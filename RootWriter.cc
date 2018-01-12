#include "tensorflow/core/framework/reader_base.h"
#include "tensorflow/core/framework/op.h"



using namespace tensorflow;


REGISTER_OP("RootWriter")
    .Input("input: float32")
    .Attr("branches: list(string)")
    .Attr("treename: string")
    .Attr("filename: string")
    //.Output("output: int32")
    .SetIsStateful()
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) 
    {
        //shape_inference::ShapeHandle s = c->MakeShape({1});
        //c->set_output(0,s);
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

class RootWriterOp:
    public OpKernel
{
    private:
        static mutex globalMutexForROOT_; //protects ROOT
        mutex localMutex_; //protects class members
        std::unique_ptr<TFile> outputFile_;
        TTree* tree_;
        std::vector<std::pair<std::string,float>> branches_;
        
    public:
        explicit RootWriterOp(OpKernelConstruction* context): 
            OpKernel(context),
            outputFile_(nullptr)
        {
            
            std::vector<string> branchNames;
            OP_REQUIRES_OK(
                context,
                context->GetAttr("branches",&branchNames)
            );
            string tree_name;
            OP_REQUIRES_OK(
                context,
                context->GetAttr("treename",&tree_name)
            );
            string file_name;
            OP_REQUIRES_OK(
                context,
                context->GetAttr("filename",&file_name)
            );
            
            mutex_lock rootLock(globalMutexForROOT_);
            
            outputFile_.reset(new TFile(file_name.c_str(),"RECREATE"));
            tree_ = new TTree(tree_name.c_str(),tree_name.c_str());
            tree_->SetDirectory(outputFile_.get());
            
            for (auto& name: branchNames)
            {
                auto p = std::find(name.begin(),name.end(),'/');
                string branchName = name;
                if (p!=name.end())
                {
                    branchName = string(name.begin(),p);
                }
                /*
                if (std::find(branches_.begin(),branches_.end(),[&branchName](const auto& elem)->bool{elem.first==branchName})!=branches_.end())
                {
                }
                */
                auto branch = std::pair<std::string,float>(branchName,0.f);
                //std::cout<<name<<" = default"<<std::endl;
                tree_->Branch(branch.first.c_str(),&branch.second);
                branches_.emplace_back(std::move(branch));
            }
        }
        
        virtual ~RootWriterOp()
        {
            mutex_lock rootLock(globalMutexForROOT_);
            outputFile_->cd(); //set gDirectory
            tree_->Write();
            outputFile_->Close();
            branches_.clear();
        }

        void Compute(OpKernelContext* context)
        {
            //std::cout<<"compute writer"<<std::endl;
            
            mutex_lock localLock(localMutex_);
            
            const Tensor& input_tensor = context->input(0);
            auto input = input_tensor.flat<float>();
            if (input.size()!=branches_.size())
            {
                throw std::runtime_error("Mismatching tensor ("+std::to_string(input.size())+")  <-> branch length ("+std::to_string(branches_.size())+")");
            }
            for (unsigned int i = 0; i < input.size(); ++i)
            {
                //std::cout<<"writing "<<branches_[i]->name()<<" = "<<input(i)<<std::endl;
                branches_[i].second=input(i);
            }
            mutex_lock rootLock(globalMutexForROOT_);
            outputFile_->cd(); //set gDirectory
            tree_->Fill();
        }
};

//mutex RootWriterOp::globalMutexForROOT_;

REGISTER_KERNEL_BUILDER(Name("RootWriter").Device(DEVICE_CPU),RootWriterOp);

