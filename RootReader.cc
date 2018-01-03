#include "tensorflow/core/framework/reader_base.h"
#include "tensorflow/core/framework/op.h"

#include <regex>

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
    .Attr("naninf: int = 0")
    .SetIsStateful()
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) 
    {
        std::vector<string> branchNames;
        TF_RETURN_IF_ERROR(c->GetAttr("branches",&branchNames));
        unsigned int size = 0;
        for (auto name: branchNames)
        {
            static std::regex syntaxRegex("[A-Za-z_0-9]+\\[[A-Za-z_0-9]+,[0-9]+\\]");
            if (not std::regex_match(name.begin(),name.end(),syntaxRegex))
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
        shape_inference::ShapeHandle s = c->MakeShape({c->MakeDim(size)});
        c->set_output(0, s);
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
        template<typename T>
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
                
                static T resetNanOrInf(const T& v, const T& reset)
                {
                    if (std::isnan(v) or std::isinf(v))
                    {
                        return reset;
                    }
                    return v;
                }
                
                virtual void setBranchAddress(TTree* tree) = 0;
                virtual unsigned int fillTensor(typename TTypes<T>::Flat& flatTensor, unsigned int index, const T& reset) const = 0;
        };
        
        template<typename T>
        class SingleBranch:
            public Branch<T>
        {
            private:
                T value_;
            public:
                SingleBranch(const string& name):
                    Branch<T>(name)
                {
                }
                
                inline const T& value() const
                {
                    return value_;
                }
                
                virtual void setBranchAddress(TTree* tree)
                {
                    tree->SetBranchAddress(Branch<T>::name().c_str(),&value_);
                }
                virtual unsigned int fillTensor(typename TTypes<T>::Flat& flatTensor,unsigned int index, const T& reset) const
                {
                    flatTensor(index)=Branch<T>::resetNanOrInf(value_,reset);
                    return index+1;
                }
        }; 
        
        template<typename T>
        class ArrayBranch:
            public Branch<T>
        {
            private:
                T* values_;
                unsigned int size_;
                std::shared_ptr<SingleBranch<unsigned int>> length_;
                 
            public:
                ArrayBranch(const string& name, std::shared_ptr<SingleBranch<unsigned int>>& length, unsigned int size):
                    Branch<T>(name),
                    values_(new T(size)),
                    length_(length),
                    size_(size)
                {
                }
                
                inline const T& value(unsigned int index) const
                {
                    if (index>=size_)
                    {
                        throw std::runtime_error("Array index out-of-range");
                    }
                    return values_[index];
                }
                
                virtual ~ArrayBranch()
                {
                    delete[] values_;
                }
                
                virtual void setBranchAddress(TTree* tree)
                {
                    tree->SetBranchAddress(Branch<T>::name().c_str(),values_);
                }
                virtual unsigned int fillTensor(typename TTypes<T>::Flat& flatTensor, unsigned int index, const T& reset) const
                {
                    //std::cout<<"length="<<length_->value()<<std::endl;
                    for (unsigned int i = 0; i < std::min(length_->value(),size_); ++i)
                    {
                        //std::cout<<i<<": "<<values_[i]<<std::endl;
                        flatTensor(index+i)=Branch<T>::resetNanOrInf(values_[i],reset);
                    }
                    for (unsigned int i = std::min(length_->value(),size_); i < size_; ++i)
                    {
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
        std::unordered_map<string,std::shared_ptr<SingleBranch<unsigned int>>> arrayLengths_;
        size_t currentEntry_;
        
        int naninf_;
        
        unsigned int size_;
        
    public:
        explicit RootReaderOp(OpKernelConstruction* context): 
            OpKernel(context),
            inputFile_(nullptr),
            currentEntry_(0),
            naninf_(0),
            size_(0)
        {
            std::vector<string> branchNames;
            OP_REQUIRES_OK(
                context,
                context->GetAttr("branches",&branchNames)
            );
            OP_REQUIRES_OK(
                context,
                context->GetAttr("naninf",&naninf_)
            );
            
            for (auto& name: branchNames)
            {
                static std::regex syntaxRegex("[A-Za-z_0-9]+\\[[A-Za-z_0-9]+,[0-9]+\\]");
                if (not std::regex_match(name.begin(),name.end(),syntaxRegex))
                {
                    branches_.emplace_back(std::make_shared<SingleBranch<float>>(name));
                    size_+=1;
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
                        arrayLengths_[lengthName]=std::make_shared<SingleBranch<unsigned int>>(lengthName);
                    }
                    branches_.emplace_back(
                        std::make_shared<ArrayBranch<float>>(branchName,arrayLengths_[lengthName],size)
                    );
                }
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
                if (!context->status().ok())
                {
                    return; //status is bad when queue is closed, so no more files -> training has finished
                }
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
                for (auto& branch: branches_)
                {
                    branch->setBranchAddress(tree_);
                }
                for (auto& branchPair: arrayLengths_)
                {
                    branchPair.second->setBranchAddress(tree_);
                }
                
            }
            tree_->GetEntry(currentEntry_);
            //std::cout<<"entry: "<<currentEntry_<<std::endl;

            Tensor* output_tensor = nullptr;
            TensorShape shape;
            //shape.AddDim(branches_.size());
            shape.AddDim(size_);
            OP_REQUIRES_OK(context, context->allocate_output("out", shape,&output_tensor));
            
            auto output_flat = output_tensor->flat<float>();
            /*
            for (unsigned int i = 0; i < _branches.size(); ++i)
            {
                auto& varPair = _branches.at(i);
                //std::cout<<"  "<<varPair.first<<": "<<varPair.second<<std::endl;
                output_flat(i) = resetNanOrInf(varPair.second,naninf_);
            }
            */
            unsigned int index = 0;
            for (auto& branch: branches_)
            {
                index = branch->fillTensor(output_flat,index,naninf_);
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

