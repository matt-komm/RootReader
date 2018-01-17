#include "tensorflow/core/framework/reader_base.h"
#include "tensorflow/core/framework/op.h"

#include <regex>

using namespace tensorflow;

//TODO: somehow need to parse pt for weight evaluation!!!

REGISTER_OP("ClassificationWeights")
    .Input("labels: float32")
    .Input("input: float32")
    .Output("out: float32")
    .Attr("rootfile: string")
    .Attr("histnames: list(string)")
    .Attr("varindex: int")
    .SetIsStateful()
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) 
    {
        tensorflow::shape_inference::ShapeHandle label_shape =  c->input(0);
        int batch_dim = c->Value(c->Dim(label_shape,0));
        //int label_length = c->Value(c->Dim(label_shape,1));
        std::vector<std::string> histNames;
        
        //std::cout<<c->Rank(label_shape)<<","<<batch_dim<<","<<label_length<<std::endl;
        TF_RETURN_IF_ERROR(c->GetAttr("histnames",&histNames));
        /*
        if (label_length!=histNames.size())
        {
            throw std::runtime_error("Labels ("+std::to_string(histNames.size())+") need to be of same size as tensor ("+std::to_string(label_length)+")");
        }
        */
        shape_inference::ShapeHandle s = c->MakeShape({batch_dim});
        c->set_output(0,s);
        return Status::OK();
    })
    .Doc(R"doc(A Reader that outputs the lines of a file delimited by '\n'.)doc");

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

#include "TH1F.h"
#include "TFile.h"


class ClassificationWeightsOp:
    public OpKernel
{

    private:
        std::string filePath;
        std::vector<std::string> histNames;
        bool transpose_;
        std::vector<TH1F> hists;
        int varIndex;
    public:
        explicit ClassificationWeightsOp(OpKernelConstruction* context): 
            OpKernel(context)
        {
            OP_REQUIRES_OK(
                context,
                context->GetAttr("rootfile",&filePath)
            );
            OP_REQUIRES_OK(
                context,
                context->GetAttr("histnames",&histNames)
            );
            OP_REQUIRES_OK(
                context,
                context->GetAttr("varindex",&varIndex)
            );
            TFile rootFile(filePath.c_str());
            if (not rootFile.IsOpen ())
            {
                throw std::runtime_error("Root file '"+filePath+"' cannot be opened");
            }
            for (auto histName: histNames)
            {
                TH1F* hist = dynamic_cast<TH1F*>(rootFile.Get(histName.c_str()));
                if (not hist)
                {
                    throw std::runtime_error("Cannot find hist '"+histName+"' in file '"+filePath+"'");
                }
                hist->SetDirectory(0);
                hists.emplace_back(*hist);
            }
        }
        
        
        virtual ~ClassificationWeightsOp()
        { 
        }
        
        float computeWeight(int classIndex, float value)
        {
            TH1& hist = hists[classIndex];
            int bin = hist.FindBin(value);
            if (bin==0 or bin>=hist.GetNbinsX()+1)
            {
                return 0;
            }
            return hist.GetBinContent(bin);
        }

        void Compute(OpKernelContext* context)
        {
            const Tensor& label_tensor = context->input(0);
            auto label = label_tensor.flat<float>();
            long num_batches = label_tensor.dim_size(0);
            long label_length = label_tensor.dim_size(1);
            if (label_length!=hists.size())
            {
                throw std::runtime_error("Labels ("+std::to_string(hists.size())+") need to be of same size as tensor ("+std::to_string(label_length)+")");
            }

            const Tensor& value_tensor = context->input(1);
            auto value = value_tensor.flat<float>();
            long value_size = value_tensor.dim_size(1);

            Tensor* output_tensor = nullptr;
            TensorShape shape;
            shape.AddDim(num_batches);
            OP_REQUIRES_OK(context, context->allocate_output("out", shape,&output_tensor));
            auto output = output_tensor->flat<float>();

            for (unsigned int ibatch = 0; ibatch < num_batches; ++ibatch)
            {
                int class_index = -1;
                for (unsigned int i = 0; i < label_length; ++i)
                { 
                    if (label(ibatch*label_length+i)>0.5)
                    {
                        class_index = i;
                        break;
                    }
                }
                if (class_index<0) throw std::runtime_error("labels tensor needs to be one-hot encoded");
                float varValue = value(ibatch*value_size+varIndex);
                
                output(ibatch) = computeWeight(class_index,varValue);
            }
        }  
};

REGISTER_KERNEL_BUILDER(Name("ClassificationWeights").Device(DEVICE_CPU),ClassificationWeightsOp);

