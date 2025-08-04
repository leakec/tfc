#include "BF.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

template <typename T> void add1DInit(auto &c) {
    c.def(py::init([](double x0, double xf, py::array_t<int> nC, int min) {
              if (nC.ndim() != 1) {
                  throw py::value_error("The \"nC\" input array must be 1-dimensional.");
              }
              return std::make_unique<T>(x0, xf, nC.data(), nC.size(), min);
          }),
          py::arg("x0"),
          py::arg("xf"),
          py::arg("nC"),
          py::arg("min"),
          R"(
            Constructor.

            Parameters:
            x0: Start of domain
            xf: End of domain
            nC: Array of indices to remove (1D numpy array)
            min: Number of basis functions to use
        )");
}

template <typename T> void addNdInit(auto &c) {
    c.def(py::init([](py::array_t<double> x0, py::array_t<double> xf, py::array_t<int> nC, int min) {
              if (x0.ndim() != 1) {
                  throw py::value_error("The \"x0\" input array must be 1-dimensional.");
              }
              if (xf.ndim() != 1) {
                  throw py::value_error("The \"xf\" input array must be 1-dimensional.");
              }
              if (nC.ndim() != 2) {
                  throw py::value_error("The \"nC\" input array must be 2-dimensional.");
              }
              return std::make_unique<T>(
                  x0.data(), x0.size(), xf.data(), xf.size(), nC.data(), nC.shape()[0], nC.shape()[1], min);
          }),
          py::arg("x0"),
          py::arg("xf"),
          py::arg("nC"),
          py::arg("min"),
          R"(
            Constructor.

            Parameters:
            x0: Start of domain
            xf: End of domain
            nC: Array of indices to remove (2D numpy array)
            min: Number of basis functions to use
        )");
}

template <typename T> void addNdElmInit(auto &c) {
    c.def(py::init([](py::array_t<double> x0, py::array_t<double> xf, py::array_t<int> nC, int min) {
              if (x0.ndim() != 1) {
                  throw py::value_error("The \"x0\" input array must be 1-dimensional.");
              }
              if (xf.ndim() != 1) {
                  throw py::value_error("The \"xf\" input array must be 1-dimensional.");
              }
              if (nC.ndim() != 1) {
                  throw py::value_error("The \"nC\" input array must be 1-dimensional.");
              }
              return std::make_unique<T>(x0.data(), x0.size(), xf.data(), xf.size(), nC.data(), nC.size(), min);
          }),
          py::arg("x0"),
          py::arg("xf"),
          py::arg("nC"),
          py::arg("min"),
          R"(
            Constructor.

            Parameters:
            x0: Start of domain (1D numpy array)
            xf: End of domain (1D numpy array)
            nC: Array of indices to remove (1D numpy array)
            min: Number of basis functions to use
        )");
}

PYBIND11_MODULE(BF, m) {

    py::class_<BasisFunc>(m, "BasisFunc")
        .def_readwrite("z0", &BasisFunc::z0)
        .def_readwrite("x0", &BasisFunc::x0)
        .def_readwrite("c", &BasisFunc::c)
        .def_readwrite("m", &BasisFunc::m)
        .def_readwrite("numC", &BasisFunc::numC)
        .def_readwrite("identifier", &BasisFunc::identifier)
        .def_property_readonly("xlaCapsule",
                               [](BasisFunc &self) {
                                   py::object capsule = py::reinterpret_borrow<py::object>(self.xlaCapsule);
                                   return capsule;
                               })
// GPU Capsule (only if available)
#ifdef HAS_CUDA
        .def_property_readonly("xlaGpuCapsule",
                               [](BasisFunc &self) { return py::reinterpret_borrow<py::object>(self.xlaGpuCapsule); })
#else
        .def_property_readonly("xlaGpuCapsule", [](BasisFunc&) {
            return "CUDA NOT FOUND, GPU NOT IMPLEMENTED.";
        })
#endif
        // Methods
        .def(
            "H",
            [](BasisFunc &self, py::array_t<double, py::array::c_style | py::array::forcecast> x, int d, bool full) {
                if (x.ndim() != 1) {
                    throw py::value_error("The \"x\" input array must be 1-dimensional.");
                }
                int n = x.size();
                int nOut = 0;
                int mOut = 0;
                double *F = nullptr;
                self.H(x.data(), n, d, &nOut, &mOut, &F, full);

                // Wrap data in a py::capsule to ensure it gets deleted
                auto capsule = py::capsule(F, [](void *f) {
                    double *d = reinterpret_cast<double *>(f);
                    free(d);
                });

                return py::array_t<double>({nOut, mOut}, F, capsule);
            },
            py::arg("x"),
            py::arg("d"),
            py::arg("full"),
            R"(
                Compute basis function matrix.

                Parameters:
                x: Points (1D numpy array)
                d: Derivative order
                full: Whether to return full matrix (not removing nC columns)

                Returns:
                mOut x nOut NumPy array.
            )");

    auto PyCP = py::class_<CP, BasisFunc>(m, "CP", py::multiple_inheritance());
    add1DInit<CP>(PyCP);

    auto PyLeP = py::class_<LeP, BasisFunc>(m, "LeP", py::multiple_inheritance());
    add1DInit<LeP>(PyLeP);

    auto PyLaP = py::class_<LaP, BasisFunc>(m, "LaP", py::multiple_inheritance());
    add1DInit<LaP>(PyLaP);

    auto PyHoPpro = py::class_<HoPpro, BasisFunc>(m, "HoPpro", py::multiple_inheritance());
    add1DInit<HoPpro>(PyHoPpro);

    auto PyHoPphy = py::class_<HoPphy, BasisFunc>(m, "HoPphy", py::multiple_inheritance());
    add1DInit<HoPphy>(PyHoPphy);

    auto PyFS = py::class_<FS, BasisFunc>(m, "FS", py::multiple_inheritance());
    add1DInit<FS>(PyFS);

    py::class_<ELM, BasisFunc>(m, "ELM")
        .def_property(
            "b",
            [](ELM &self) {
                double *data = nullptr;
                int nOut;
                self.getB(&data, &nOut);

                auto capsule = py::capsule(data, [](void *f) {
                    double *d = reinterpret_cast<double *>(f);
                    free(d);
                });
                return py::array_t<double>(self.m, data, capsule);
            },
            [](ELM &self, py::array_t<double> b) { self.setB(b.data(), b.size()); })
        .def_property(
            "w",
            [](ELM &self) {
                double *data = nullptr;
                int nOut;
                self.getW(&data, &nOut);

                auto capsule = py::capsule(data, [](void *f) {
                    double *d = reinterpret_cast<double *>(f);
                    free(d);
                });
                return py::array_t<double>(self.m, data, capsule);
            },
            [](ELM &self, py::array_t<double> w) { self.setW(w.data(), w.size()); });

    auto PyELMSigmoid = py::class_<ELMSigmoid, ELM>(m, "ELMSigmoid");
    add1DInit<ELMSigmoid>(PyELMSigmoid);

    auto PyELMReLU = py::class_<ELMReLU, ELM>(m, "ELMReLU");
    add1DInit<ELMReLU>(PyELMReLU);

    auto PyELMTanh = py::class_<ELMTanh, ELM>(m, "ELMTanh");
    add1DInit<ELMTanh>(PyELMTanh);

    auto PyELMSin = py::class_<ELMSin, ELM>(m, "ELMSin");
    add1DInit<ELMSin>(PyELMSin);

    auto PyELMSwish = py::class_<ELMSwish, ELM>(m, "ELMSwish");
    add1DInit<ELMSwish>(PyELMSwish);

    // TODO: Finish members and add methods.
    py::class_<nBasisFunc, BasisFunc>(m, "nBasisFunc", py::multiple_inheritance())
        .def_readwrite("z0", &nBasisFunc::z0)
        .def_readwrite("zf", &nBasisFunc::zf)
        .def_readwrite("dim", &nBasisFunc::dim)
        .def_property(
            "c",
            [](nBasisFunc &self) {
                // Return c, and ensure the nBasisFunc stays around as long as c does.
                return py::array_t<double>(self.dim, self.c, py::cast(self));
            },
            [](nBasisFunc &self, py::array_t<double, py::array::c_style | py::array::forcecast> c) {
                if (c.ndim() != 1) {
                    throw py::value_error("The \"c\" input array must be 1-dimensional.");
                }
                if (c.size() != self.dim) {
                    std::stringstream ss;
                    ss << "The \"c\" input array must be size " << self.dim << ", but got size " << c.size() << "."
                       << std::endl;
                    throw py::value_error(ss.str());
                }
            })
        .def_readwrite("numBasisFunc", &nBasisFunc::numBasisFunc)
        .def_readwrite("numBasisFuncFull", &nBasisFunc::numBasisFuncFull)
        .def(
            "H",
            [](nBasisFunc &self,
               py::array_t<double, py::array::c_style | py::array::forcecast> x,
               py::array_t<int, py::array::c_style | py::array::forcecast> d,
               bool full) {
                if (x.ndim() != 2) {
                    throw py::value_error("The \"x\" input array must be 2-dimensional.");
                }
                if (d.ndim() != 1) {
                    throw py::value_error("The \"d\" input array must be 1-dimensional.");
                }
                int nOut = 0;
                int mOut = 0;
                double *F = nullptr;
                self.H(x.data(), x.shape()[0], x.shape()[1], d.data(), d.shape()[0], &nOut, &mOut, &F, full);

                // Wrap data in a py::capsule to ensure it gets deleted
                auto capsule = py::capsule(F, [](void *f) {
                    double *d = reinterpret_cast<double *>(f);
                    free(d);
                });

                return py::array_t<double>({nOut, mOut}, F, capsule);
            },
            py::arg("x"),
            py::arg("d"),
            py::arg("full"),
            R"(
                Compute basis function matrix.

                Parameters:
                x: Points (1D numpy array)
                d: Derivative order
                full: Whether to return full matrix (not removing nC columns)

                Returns:
                mOut x nOut NumPy array.
            )");

    auto PynCP = py::class_<nCP, nBasisFunc>(m, "nCP");
    addNdInit<nCP>(PynCP);

    auto PynLeP = py::class_<nLeP, nBasisFunc>(m, "nLeP");
    addNdInit<nLeP>(PynLeP);

    auto PynFS = py::class_<nFS, nBasisFunc>(m, "nFS");
    addNdInit<nFS>(PynFS);

    py::class_<nELM, nBasisFunc>(m, "nELM")
        .def_property(
            "b",
            [](nELM &self) {
                double *data = nullptr;
                int nOut;
                self.getB(&data, &nOut);

                auto capsule = py::capsule(data, [](void *f) {
                    double *d = reinterpret_cast<double *>(f);
                    free(d);
                });
                return py::array_t<double>(self.m, data, capsule);
            },
            [](nELM &self, py::array_t<double> b) { self.setB(b.data(), b.size()); })
        .def_property(
            "w",
            [](nELM &self) {
                double *data = nullptr;
                int nOut;
                int dimOut;
                self.getW(&dimOut, &nOut, &data);

                auto capsule = py::capsule(data, [](void *f) {
                    double *d = reinterpret_cast<double *>(f);
                    free(d);
                });
                return py::array_t<double>({dimOut, nOut}, data, capsule);
            },
            [](nELM &self, py::array_t<double, py::array::c_style | py::array::forcecast> w) {
                if (w.ndim() != 2) {
                    throw py::value_error("The \"w\" input array must be 2-dimensional.");
                }
                self.setW(w.data(), w.shape()[0], w.shape()[1]);
            });

    auto PynELMSigmoid = py::class_<nELMSigmoid, nELM>(m, "nELMSigmoid");
    addNdElmInit<nELMSigmoid>(PynELMSigmoid);

    auto PynELMTanh = py::class_<nELMTanh, nELM>(m, "nELMTanh");
    addNdElmInit<nELMTanh>(PynELMTanh);

    auto PynELMSin = py::class_<nELMSin, nELM>(m, "nELMSin");
    addNdElmInit<nELMSin>(PynELMSin);

    auto PynELMSwish = py::class_<nELMSwish, nELM>(m, "nELMSwish");
    addNdElmInit<nELMSwish>(PynELMSwish);

    auto PynELMReLU = py::class_<nELMReLU, nELM>(m, "nELMReLU");
    addNdElmInit<nELMReLU>(PynELMReLU);
}
