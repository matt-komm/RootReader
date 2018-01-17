#include "tensorflow/core/framework/reader_base.h"
#include "tensorflow/core/framework/op.h"

#include <regex>

using namespace tensorflow;

//TODO: somehow need to parse pt for weight evaluation!!!

REGISTER_OP("ClassificationWeights")
    .Input("in: int32")
    .Output("out: float32")
    .Attr("rootfile: string")
    .Attr("histnames: list(string)")
    .SetIsStateful()
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) 
    {
        int batch_dim = c->Value(c->Dim(c->input(0),0));
        int label_length = c->Value(c->Dim(c->input(0),1));
        std::vector<std::string> histNames;
        TF_RETURN_IF_ERROR(c->GetAttr("histnames",&histNames));
        if (label_length!=histNames.size())
        {
            throw std::runtime_error("Labels need to be of same size as tensor");
        }
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

        void Compute(OpKernelContext* context)
        {
            const Tensor& input_tensor = context->input(0);
            auto input = input_tensor.flat<int>();
            long num_batches = input_tensor.dim_size(0);
            long label_length = input_tensor.dim_size(1);

            Tensor* output_tensor = nullptr;
            TensorShape shape;
            shape.AddDim(num_batches);
            shape.AddDim(label_length);
            OP_REQUIRES_OK(context, context->allocate_output("out", shape,&output_tensor));
            auto output = output_tensor->flat<float>();

            for (unsigned int ibatch = 0; ibatch < num_batches; ++ibatch)
            {
                int label_index = 0;
                for (unsigned int i = 0; i < label_length; ++i)
                { 
                    label_index = input(ibatch*label_length+i);
                    if (label_index>0) break;
                }
                
            }
        }  
};

REGISTER_KERNEL_BUILDER(Name("ClassificationWeights").Device(DEVICE_CPU),ClassificationWeightsOp);

