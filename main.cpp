#include <torch/script.h> // One-stop header.
#include <torch/data/dataloader.h>
#include <iostream>
#include <memory>
#include<math.h>
int main(int argc, const char* argv[]) {

  torch::jit::script::Module module;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    module = torch::jit::load(argv[1]);
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
  }
  auto single_dataset = torch::data::DataLoader('train');
  std::cout<<single_dataset<<"\n";
  std::cout<<"Enter numbers: ";
  float x1,x2,x3,x4;
  std::cin>>x1,x2,x3,x4;
  torch::Tensor tharray = torch::tensor({{x1,x2,x3,x4}});
  //tharray.to(torch::kLong);


  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(tharray);

  //Execute the model and turn its output into a tensor.
  at::Tensor output = module.forward(inputs).toTensor();
  std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';

  float max = output[0][0].item<float_t>();
  int index = 0;
  // Finding winner
  for(int i = 1; i < 3; i++){
    if (max < output[0][i].item<float_t>()){
        max = output[0][i].item<float_t>();
        index = i;
    }

  }

  std::string flower_labels[3] = {"Iris","Virginica","Versicolor"};
  // Printing probability 
  std::cout<<"Model predicted: "<<flower_labels[index]<<"\n";
  printf("Confidence level: %.2f %% \n",100*exp(output[0][0].item<float_t>()));
  return 0;
}