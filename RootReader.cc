#include "tensorflow/core/framework/reader_base.h"
#include "tensorflow/core/framework/op.h"



using namespace tensorflow;

//NOTE: regex is experimental in gcc 4.9 and below
//TODO: define proper syntax and parsing e.g. support
//  <name>; <name>/<type>; <name>[<num>,<max>]; <name>[<num>,<max>]/<type>
//TODO (optional): add selections, add support for TSelector
namespace syntax_test
{
    static bool isArray(const string& s)
    {
        auto p1 = std::find(s.begin(),s.end(),'[');
        auto p2 = std::find(s.begin(),s.end(),',');
        auto p3 = std::find(s.begin(),s.end(),']');
        return p1!=s.end() and p2!=s.end() and p3!=s.end() and p1<p2 and p2<p3;;
    }
}

REGISTER_OP("RootReader")
    .Input("queue_handle: resource")
    .Attr("branches: list(string)")
    .Attr("treename: string")
    .Attr("naninf: int = 0")
    .Attr("batch: int = 1")
    .Output("out: float32")
    .Output("num: int32")
    .SetIsStateful()
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) 
    {
        std::vector<string> branchNames;
        TF_RETURN_IF_ERROR(c->GetAttr("branches",&branchNames));
        unsigned int size = 0;
        for (auto name: branchNames)
        {
            
            if (not syntax_test::isArray(name))
            {
                size+=1;
            }
            else
            {
                auto p1 = std::find(name.begin(),name.end(),'[');
                auto p2 = std::find(p1,name.end(),',');
                auto p3 = std::find(p2,name.end(),']');
                
                size += std::stol(std::string(p2+1,p3));
            }
        }
        //shape_inference::ShapeHandle s = c->MakeShape({c->MakeDim(branchNames.size())});
        shape_inference::ShapeHandle s1 = c->MakeShape({-1,c->MakeDim(size)});
        c->set_output(0, s1);
        
        shape_inference::ShapeHandle s2 = c->MakeShape({-1,1});
        c->set_output(1, s2);
        return Status::OK();
    })
    .Doc(R"doc(A Reader that outputs the lines of a file delimited by '\n'.)doc");

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"


#include "TFile.h"
#include "TTree.h"

#include <vector>
#include <memory>

#include <chrono>
#include <thread>

class RootReaderOp:
    public OpKernel
{

    public:
        template<typename OUT>
        class Branch
        {
            protected:
                string name_;
            public:
                Branch(const string& name):
                    name_(name)
                {
                }
                
                inline const string name() const
                {
                    return name_;
                }
                
                static OUT resetNanOrInf(const OUT& v, const OUT& reset)
                {
                    if (std::isnan(v) or std::isinf(v))
                    {
                        return reset;
                    }
                    return v;
                }
                
                virtual void setBranchAddress(TTree* tree) = 0;
                virtual unsigned int fillTensor(typename TTypes<OUT>::Flat& flatTensor, unsigned int index, const OUT& reset) const = 0;
        };
        
        template<typename IN, typename OUT=IN>
        class SingleBranch:
            public Branch<OUT>
        {
            private:
                IN value_;
            public:
                SingleBranch(const string& name):
                    Branch<OUT>(name)
                {
                }
                
                inline const IN& value() const
                {
                    return value_;
                }
                
                virtual void setBranchAddress(TTree* tree)
                {
                    if(tree->SetBranchAddress(Branch<OUT>::name().c_str(),&value_)<0)
                    {
                        throw std::runtime_error("No branch with name '"+Branch<OUT>::name()+"' in tree");
                    }
                }
                virtual unsigned int fillTensor(typename TTypes<OUT>::Flat& flatTensor,unsigned int index, const OUT& reset) const
                {
                    flatTensor(index)=Branch<OUT>::resetNanOrInf(value_,reset);
                    //std::cout<<index<<": "<<Branch<OUT>::name()<<"="<<flatTensor(index)<<std::endl;
                    return index+1;
                }
        }; 
        
        template<typename IN, typename OUT=IN>
        class ArrayBranch:
            public Branch<OUT>
        {
            private:
                alignas(16) IN values_[50]; //there is some odd bug when using dynamic allocated arrays and root
                unsigned int size_;
                std::shared_ptr<SingleBranch<unsigned int,OUT>> length_;
                 
            public:
                ArrayBranch(const string& name, std::shared_ptr<SingleBranch<unsigned int,OUT>>& length, unsigned int size):
                    Branch<OUT>(name),
                    length_(length),
                    size_(size)
                {
                }
                
                inline const IN& value(unsigned int index) const
                {
                    if (index>=size_)
                    {
                        throw std::runtime_error("Array index out-of-range");
                    }
                    return values_[index];
                }
                
                virtual ~ArrayBranch()
                {
                    //delete[] values_;
                }
                
                //error codes: https://root.cern.ch/doc/master/classTTree.html#a1a48bf75621868a514741b27252cad96
                virtual void setBranchAddress(TTree* tree)
                {
                    if(tree->SetBranchAddress(Branch<OUT>::name().c_str(),values_)<0)
                    {
                        throw std::runtime_error("No branch with name '"+Branch<OUT>::name()+"' in tree");
                    }
                }
                virtual unsigned int fillTensor(typename TTypes<OUT>::Flat& flatTensor, unsigned int index, const OUT& reset) const
                {
                    //std::cout<<Branch<OUT>::name()<<", length="<<length_->value()<<std::endl;
                    for (unsigned int i = 0; i < std::min(length_->value(),size_); ++i)
                    {
                        //std::cout<<(index+i)<<": "<<values_[i]<<std::endl;
                        flatTensor(index+i)=Branch<OUT>::resetNanOrInf(values_[i],reset);
                    }
                    for (unsigned int i = std::min(length_->value(),size_); i < size_; ++i)
                    {
                        //std::cout<<(index+i)<<": padded"<<std::endl;
                        flatTensor(index+i) = 0; //zero padding
                    }
                    
                    return index+size_;
                }
        };
        
    private:
        static mutex globalMutexForROOT_; //protects ROOT
        mutex localMutex_; //protects class members
        std::unique_ptr<TFile> inputFile_;
        TTree* tree_;
        std::vector<std::shared_ptr<Branch<float>>> branches_;
        std::unordered_map<string,std::shared_ptr<SingleBranch<unsigned int,float>>> arrayLengths_;
        size_t currentEntry_;
        
        int naninf_;
        string treename_;
        unsigned int size_;
        int nBatch_;
        unsigned int nEvents_;
        
    public:
        explicit RootReaderOp(OpKernelConstruction* context): 
            OpKernel(context),
            inputFile_(nullptr),
            currentEntry_(0),
            naninf_(0),
            size_(0),
            nBatch_(1),
            nEvents_(0)
        {
        
            mutex_lock globalLock(globalMutexForROOT_);
            
            std::vector<string> branchNames;
            OP_REQUIRES_OK(
                context,
                context->GetAttr("branches",&branchNames)
            );
            OP_REQUIRES_OK(
                context,
                context->GetAttr("treename",&treename_)
            );
            OP_REQUIRES_OK(
                context,
                context->GetAttr("naninf",&naninf_)
            );
            OP_REQUIRES_OK(
                context,
                context->GetAttr("batch",&nBatch_)
            );
            for (auto& name: branchNames)
            {
                if (not syntax_test::isArray(name))
                {
                    auto it = std::find(name.begin(),name.end(),'/');
                    if (it==name.end())
                    {
                        //std::cout<<name<<" = default"<<std::endl;
                        branches_.emplace_back(std::make_shared<SingleBranch<float>>(name));
                        size_+=1;
                    }
                    else
                    {
                        //std::cout<<name<<" = typed"<<std::endl;
                        string type(it+1,name.end());
                        if (type=="UInt_t")
                        {

                            branches_.emplace_back(std::make_shared<SingleBranch<unsigned int, float>>(string(name.begin(),it)));
                            size_+=1;
                        }
                    }
                }
                else
                {
                    auto p1 = std::find(name.begin(),name.end(),'[');
                    auto p2 = std::find(p1,name.end(),',');
                    auto p3 = std::find(p2,name.end(),']');
                    std::string branchName(name.begin(),p1);
                    std::string lengthName(p1+1,p2);
                    unsigned int size = std::stol(std::string(p2+1,p3));
                    size_+=size;
                    auto lengthBranchIt = arrayLengths_.find(lengthName);
                    //std::cout<<"branch="<<branchName<<", length="<<lengthName<<", size="<<size<<std::endl;
                    if (lengthBranchIt==arrayLengths_.end())
                    {
                        arrayLengths_[lengthName]=std::make_shared<SingleBranch<unsigned int,float>>(lengthName);
                    }
                    branches_.emplace_back(
                        std::make_shared<ArrayBranch<float>>(branchName,arrayLengths_[lengthName],size)
                    );
                }
            }
        }
        
        virtual ~RootReaderOp()
        {
            mutex_lock globalLock(globalMutexForROOT_);
            //std::cout<<"final read: "<<currentEntry_<<std::endl;
            branches_.clear();
            arrayLengths_.clear();
        }
        
        

        void Compute(OpKernelContext* context)//, DoneCallback done) override
        {
            mutex_lock localLock(localMutex_);
           
            if (not inputFile_)
            {
                QueueInterface* queue;
                OP_REQUIRES_OK(context,GetResourceFromContext(context, "queue_handle", &queue));
                
                string fileName = GetNextFilename(queue,context);
                if (!context->status().ok())
                {
                    return; //status is bad when queue is closed, so no more reduce_files -> training has finished
                }
                if (fileName.size()==0) throw std::runtime_error("Got empty filename");
                   
                //creating TFile/setting branch adresses is not thread safe
                mutex_lock rootLock(globalMutexForROOT_);
                //TODO: use TF logging and set loglevel
                //std::cout<<"opening file: "<<fileName<<std::endl;
                inputFile_.reset(new TFile(fileName.c_str()));
                currentEntry_ = 0;
                //TODO: make treename configurable
                tree_ = dynamic_cast<TTree*>(inputFile_->Get(treename_.c_str()));
                if (not tree_)
                {
                    throw std::runtime_error("Cannot get tree 'deepntuplizer/tree' from file "+fileName);
                }
                for (auto& branch: branches_)
                {
                    branch->setBranchAddress(tree_);
                }
                for (auto& branchPair: arrayLengths_)
                {
                    branchPair.second->setBranchAddress(tree_);
                }
                nEvents_ = tree_->GetEntries();
            }
            Tensor* output_tensor = nullptr;
            TensorShape shape;
            unsigned int nBatches = std::min<unsigned int>(nEvents_-currentEntry_,nBatch_);
            shape.AddDim(nBatches);
            shape.AddDim(size_);
            OP_REQUIRES_OK(context, context->allocate_output("out", shape,&output_tensor));
            
            Tensor* output_num = nullptr;
            TensorShape shape_num;
            shape_num.AddDim(nBatches);
            shape_num.AddDim(1);
            OP_REQUIRES_OK(context, context->allocate_output("num", shape_num,&output_num));
            
            auto output_flat = output_tensor->flat<float>();
            auto output_num_flat = output_num->flat<int>();
            unsigned int index = 0;
            //std::cout<<"prepare batch ..."<<std::endl;
            for (unsigned int ibatch=0; ibatch<nBatches;++ibatch)
            {
                //std::cout<<currentEntry_<<",";
                tree_->GetEntry(currentEntry_);
                output_num_flat(ibatch)=currentEntry_;
                for (auto& branch: branches_)
                {
                    index = branch->fillTensor(output_flat,index,naninf_);
                }
                ++currentEntry_;
            }
            //std::cout<<std::endl;
            if (currentEntry_>=nEvents_)
            {
                mutex_lock globalLock(globalMutexForROOT_);
                inputFile_->Close();
                inputFile_.reset(nullptr);
            }
        }
        
        string GetNextFilename(QueueInterface* queue, OpKernelContext* context) const 
        {
            //mutex_lock localLock(localMutex_); //mutex here makes deadlock for some reason
            //TODO: check core/framework/reader_base.cc for details
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

