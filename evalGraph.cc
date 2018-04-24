#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"

#include "tensorflow/core/public/session.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/io/path.h"

#include <exception>

int main()
{
    tensorflow::Status status;

    // load it
    tensorflow::GraphDef* graphDef = new tensorflow::GraphDef();
    status = tensorflow::ReadBinaryProto(tensorflow::Env::Default(), "blub.pb", graphDef);

    // check for success
    if (!status.ok())
    {
        throw std::runtime_error("InvalidGraphDef: error while loading graph def: "+status.ToString());
    }
    return 0;
}

