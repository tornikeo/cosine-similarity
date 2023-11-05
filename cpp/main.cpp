#include "npy.hpp"
#include <vector>
#include <string>

// Write example
// int main() {
//   const std::vector<double> data{1, 2, 3, 4, 5, 6};

//   npy::npy_data<double> d;
//   d.data = data;
//   d.shape = {2, 3};
//   d.fortran_order = false; // optional

//   const std::string path{"out.npy"};
//   write_npy(path, d);
// }

// Read example
int main() {
    const std::string path {"test.npy"};
    npy::npy_data d = npy::read_npy<double>(path);

    std::vector<double> data = d.data;
    std::vector<unsigned long> shape = d.shape;
    bool fortran_order = d.fortran_order;
    std::cout << d.data[0] << std::endl;
}
