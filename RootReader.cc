#include "tensorflow/core/framework/reader_base.h"
#include "tensorflow/core/framework/op.h"

using namespace tensorflow;

/*
class TTreeResource:
    public ResourceBase
{
    private:
        mutable mutex mu_;
    public:
        void read(QueueInterface* queue, OpKernelContext* context, Tensor* output, size_t nRecords)
        {
            mutex_lock lock(mu_);
            
        }
        
        
};
*/

REGISTER_OP("RootReader")
    .Input("queue_handle: resource")
    //.Input("queue_handle: Ref(string)")
    .Output("out: float32")
    .Attr("branches: list(string)")
    .SetIsStateful()
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) 
    {
        std::vector<string> branchNames;
        TF_RETURN_IF_ERROR(c->GetAttr("branches",&branchNames));
        shape_inference::ShapeHandle s = c->MakeShape({c->MakeDim(branchNames.size())});
        c->set_output(0, s);
        return Status::OK();
    })
    .Doc(R"doc(A Reader that outputs the lines of a file delimited by '\n'.)doc");

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"


#include "TFile.h"
#include "TTree.h"

#include <vector>
#include <chrono>
#include <thread>

class RootReaderOp:
    public OpKernel
{
    private:
        static mutex globalMutexForROOT_; //protects ROOT
        
        mutable mutex localMutex_; //protects class members
        std::unique_ptr<TFile> inputFile_;
        TTree* tree_;
        std::vector<std::pair<string,float>> _branches;
        size_t currentEntry_;
        
    public:
        explicit RootReaderOp(OpKernelConstruction* context): 
            OpKernel(context),
            inputFile_(nullptr),
            currentEntry_(0)
        {
            std::vector<string> branchNames;
            OP_REQUIRES_OK(
                context,
                context->GetAttr("branches",&branchNames)
            );
            for (auto& name: branchNames)
            {
                _branches.emplace_back(name,0);
            }
        }
        
        virtual ~RootReaderOp()
        {
            std::cout<<"final read: "<<currentEntry_<<std::endl;
        }

        void Compute(OpKernelContext* context)//, DoneCallback done) override
        {
            mutex_lock localLock(localMutex_);
            
            
            
            if (not inputFile_)
            {
                QueueInterface* queue;
                OP_REQUIRES_OK(context,GetResourceFromContext(context, "queue_handle", &queue));
                string fileName = GetNextFilename(queue,context);
                if (fileName.size()==0) throw std::runtime_error("Got empty filename");
                   
                //creating TFile/setting branch adresses is not thread safe
                mutex_lock rootLock(globalMutexForROOT_);
                std::cout<<"opening file: "<<fileName<<std::endl;
                inputFile_.reset(new TFile(fileName.c_str()));
                currentEntry_ = 0;
                tree_ = dynamic_cast<TTree*>(inputFile_->Get("deepntuplizer/tree"));
                if (not tree_)
                {
                    throw std::runtime_error("Cannot get tree 'deepntuplizer/tree' from file "+fileName);
                }
                for (auto& varPair: _branches)
                {
                    tree_->SetBranchAddress(varPair.first.c_str(),&varPair.second);
                }
                
            }
            tree_->GetEntry(currentEntry_);
            //std::cout<<"entry: "<<currentEntry_<<std::endl;

            Tensor* output_tensor = nullptr;
            TensorShape shape;
            shape.AddDim(_branches.size());
            OP_REQUIRES_OK(context, context->allocate_output("out", shape,&output_tensor));
            
            //not really needed?
            auto output_flat = output_tensor->flat<float>();

            for (unsigned int i = 0; i < _branches.size(); ++i)
            {
                auto& varPair = _branches.at(i);
                //std::cout<<"  "<<varPair.first<<": "<<varPair.second<<std::endl;
                output_flat(i) = varPair.second;
            }
            
            ++currentEntry_;
            
            if (currentEntry_>=tree_->GetEntries())
            {
                inputFile_->Close();
                inputFile_.reset(nullptr);
            }
        }
        
        string GetNextFilename(QueueInterface* queue, OpKernelContext* context) const 
        {
            string work;
            //if (queue->is_closed()) throw std::runtime_error("Closed queue");
            Notification n;
            queue->TryDequeue(
                context, [this, context, &n, &work](const QueueInterface::Tuple& tuple) 
                {
                    if (context->status().ok())
                    {
                        if (tuple.size() != 1) 
                        {
                            context->SetStatus(errors::InvalidArgument("Expected single component queue"));
                        } 
                        else if (tuple[0].dtype() != DT_STRING) 
                        {
                            context->SetStatus(errors::InvalidArgument("Expected queue with single string component"));
                        } 
                        else if (tuple[0].NumElements() != 1) 
                        {
                            context->SetStatus(errors::InvalidArgument("Expected to dequeue a one-element string tensor"));
                        } 
                        else 
                        {
                            work = tuple[0].flat<string>()(0);
                        }
                    }
                    n.Notify();
                }
            );
            n.WaitForNotification();
            return work;
        }   
};

mutex RootReaderOp::globalMutexForROOT_;

REGISTER_KERNEL_BUILDER(Name("RootReader").Device(DEVICE_CPU),RootReaderOp);

