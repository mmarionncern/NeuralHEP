# ROOT flags
ROOTCFLAGS   := $(shell root-config --cflags)
ROOTLIBS     := $(shell root-config --libs)

# HDF5 FLAGS
HDF5CFLAGS    = -I${HDF5CPP}/include
HDF5LIBS      = -L${HDF5CPP}/lib -lhdf5_cpp

# c++-compiler
CXX           = g++
CXXFLAGS      += -fPIC $(ROOTCFLAGS) $(HDF5CFLAGS) -I.

LIBS          = $(ROOTLIBS) $(HDF5LIBS) 

EXES = hdf2root

all: bin 

bin: $(addsuffix .e,$(EXES))

%.e: %.cc
	@echo "Linking $@"
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $< -o $@ -L. $(LIBS)
	@ln -sf $@ $(subst .e,,$@)

clean:
	@rm -f *~
	@for file in $(EXES); do (rm -f $$file.e); done
	@for file in $(EXES); do (rm -f $$file); done
