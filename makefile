# $Id: Makefile.users.in,v 1.6 2011/03/24 16:20:52 chulwoo Exp $
# Makefile.users.  Generated from Makefile.users.in by configure.
#   This is the makefile for all the commands
#   that are common for all testing applications.
#----------------------------------------------------------------------


UTIL = ~/bin
SRCDIR = /bgusr/home/jtu/cps-jtu-build/1.0/build/../cps_pp
BUILDDIR = /bgusr/home/jtu/cps-jtu-build/1.0/build
QOS = 
QOSLIB = ${QOS}/quser/gcc-lib-user///
CC = /bgsys/drivers/ppcfloor/comm/gcc/bin/mpicc
CXX = /bgsys/drivers/ppcfloor/comm/gcc/bin/mpicxx
# CC = /opt/ibmcmp/vacpp/bg/12.1/bin/xlc
# CXX = /opt/ibmcmp/vacpp/bg/12.1/bin/xlc++
AS  = /bgsys/drivers/ppcfloor/gnu-linux/bin/powerpc64-bgq-linux-as
LDFLAGS =  -lqdp -fopenmp -lgmp -L/bgusr/home/jtu/cps-jtu-build/1.0/local/lib -lqmp -L/bgusr/home/jtu/cps-jtu-build/1.0/local/lib -lqmp -lqio -llime  -L/bgusr/home/jtu/cps-jtu-build/1.0/local/lib -lxml2 -lm -L/bgusr/home/jtu/cps-jtu-build/1.0/local/bfm/lib  -lqdp -lXPathReader -lxmlWriter -lxml2 -lqio -lbfm -L/bgusr/home/jtu/cps-jtu-build/1.0/local/lib -lgsl -lgslcblas -lm -L/bgusr/home/jtu/cps-jtu-build/1.0/local/lib -lxml2 -lm -L/bgusr/home/jtu/cps-jtu-build/1.0/local/lib -lfftw3 -L/bgusr/home/jtu/qlat-build/1.0/lib -lhash-cpp

me = $(notdir $(PWD))
BIN = BGQ.x

#VPATH :=$(SRCDIR)/tests/$(me)



#
# include list for the Columbia code
#
INCLIST = -I${BUILDDIR} -I${SRCDIR}/include -I/bgusr/home/jtu/cps-jtu-build/1.0/local/bfm/include  -I/bgusr/home/jtu/cps-jtu-build/1.0/local/include    -I/bgusr/home/jtu/cps-jtu-build/1.0/local/include/libxml2  -I/bgusr/home/jtu/cps-jtu-build/1.0/local/include -I/bgusr/home/jtu/cps-jtu-build/1.0/local/include/eigen3 -I/bgusr/home/jtu/cps-jtu-build/1.0/local/include/libxml2 -I/bgusr/home/jtu/cps-jtu-build/1.0/local/include -I/bgusr/home/jtu/Qlattice/ -I/bgusr/home/jtu/qlat-build/1.0/include -I/bgusr/home/jtu/RngState-cc -I/bgusr/home/jtu/Timer

CFLAGS= -g -O2 -fopenmp -O2 -Wall -std=c++0x -fno-strict-aliasing
CXXFLAGS=  -fopenmp -O2 -Wall -std=c++0x -fno-strict-aliasing
ASFLAGS= 
DFLAGS +=  -DUSE_OMP -DVEC_INLINE -DGMP -DUSE_QMP -DUSE_QIO -DUSE_BFM -DUSE_BFM_MINV -DUSE_BFM_TM -DUSE_HDCG -DUSE_GSL -DUSE_FFTW

#
# Libraries for the Columbia code
#
# (These are for the scalar version of the code)
#
#

.PHONY: cps clean


LIBLIST =\
  $(BUILDDIR)/cps.a \

#
#  targets
#


all: clean $(BIN)

run: 
	$(UTIL)/run-job.sh ./run.sh

see:
	tail logs/latest -f

.SUFFIXES:
.SUFFIXES:  .o .C .S .c

CSRC :=$(wildcard *.c)
CCSRC :=$(wildcard *.C)
SSRC :=$(wildcard *.S)

COBJ=$(CSRC:.c=.o)
CCOBJ=$(CCSRC:.C=.o)
SOBJ=$(SSRC:.S=.o)

OBJS_SRC = $(SOBJ) $(CCOBJ) $(COBJ)
OBJS := $(notdir $(OBJS_SRC))

$(BIN):  $(OBJS) $(LIBLIST)
	@echo OBJS = $(OBJS)
	$(CXX) $(OBJS) $(LIBLIST) $(LDFLAGS) -o $(BIN)

.c.o:
	$(CC) -o $@ $(CFLAGS) $(DFLAGS) -c $(INCLIST) $<
.C.o:
	$(CXX) -o $@ $(CXXFLAGS) $(DFLAGS) -c $(INCLIST) $<
.S.o:
	$(AS) -o $@ $(ASFLAGS) -c $(INCLIST) $<

cps:
	$(MAKE) -C $(BUILDDIR)

clean:
	rm -f *.dat *.o  $(BIN) || true
	rm -f ../regressions/*$(me).dat || true
	rm -f ../regressions/*$(me).checklog || true
