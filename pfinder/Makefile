################################################################################
#
# Path Finder for Xebra project
# Korea University Operating System Laboratory
#
# Author: Dojung KIM <djkim@os.korea.ac.kr>
#
################################################################################

###########################################################################
# Driver files

# GPGPU client files
GPUDIR = ../savanna_clnt
GPUCFILES = $(GPUDIR)/savanna_clnt_mod.c $(GPUDIR)/savanna_clnt_comm.c
GPUHFILES = $(GPUDIR)/savanna_clnt_mod.h

# core driver files
#CFILES = pfinder_main.c
#HFILES = pfinder.h pfinder_sysfs.h
CFILES = pfinder_main.c pfinder_ipv4.c \
		$(GPUCFILES)
HFILES = pfinder.h pfinder_sysfs.h pfinder_ipv4.h \
		$(GPUHFILES)

ifeq (,$(BUILD_KERNEL))
BUILD_KERNEL=$(shell uname -r)
endif

DRIVER_NAME=pfinder

###########################################################################
# Environment tests

# Kernel Search Path
# All the places we look for kernel source
KSP :=  /lib/modules/$(BUILD_KERNEL)/build \
        /lib/modules/$(BUILD_KERNEL)/source \
        /usr/src/linux-$(BUILD_KERNEL) \
        /usr/src/linux-$($(BUILD_KERNEL) | sed 's/-.*//') \
        /usr/src/kernel-headers-$(BUILD_KERNEL) \
        /usr/src/kernel-source-$(BUILD_KERNEL) \
        /usr/src/linux-$($(BUILD_KERNEL) | sed 's/\([0-9]*\.[0-9]*\)\..*/\1/') \
        /usr/src/linux

# prune the list down to only values that exist
# and have an include/linux sub-directory
test_dir = $(shell [ -e $(dir)/include/linux ] && echo $(dir))
KSP := $(foreach dir, $(KSP), $(test_dir))

# we will use this first valid entry in the search path
ifeq (,$(KSRC))
  KSRC := $(firstword $(KSP))
endif

ifeq (,$(KSRC))
  $(warning *** Kernel header files not in any of the expected locations.)
  $(warning *** Install the appropriate kernel development package, e.g.)
  $(error kernel-devel, for building kernel modules and try again)
else
ifeq (/lib/modules/$(BUILD_KERNEL)/source, $(KSRC))
  KOBJ :=  /lib/modules/$(BUILD_KERNEL)/build
else
  KOBJ :=  $(KSRC)
endif
endif

# Version file Search Path
VSP :=  $(KOBJ)/include/generated/utsrelease.h \
        $(KOBJ)/include/linux/utsrelease.h \
        $(KOBJ)/include/linux/version.h \
        /boot/vmlinuz.version.h

# Config file Search Path
CSP :=  $(KSRC)/include/generated/autoconf.h \
        $(KSRC)/include/linux/autoconf.h \
        /boot/vmlinuz.autoconf.h

# prune the lists down to only files that exist
test_file = $(shell [ -f $(file) ] && echo $(file))
VSP := $(foreach file, $(VSP), $(test_file))
CSP := $(foreach file, $(CSP), $(test_file))

# and use the first valid entry in the Search Paths
ifeq (,$(VERSION_FILE))
  VERSION_FILE := $(firstword $(VSP))
endif
ifeq (,$(CONFIG_FILE))
  CONFIG_FILE := $(firstword $(CSP))
endif

ifeq (,$(wildcard $(VERSION_FILE)))
  $(error Linux kernel source not configured - missing version header file)
endif

ifeq (,$(wildcard $(CONFIG_FILE)))
  $(error Linux kernel source not configured - missing autoconf.h)
endif


# pick a compiler
ifneq (,$(findstring egcs-2.91.66, $(shell cat /proc/version)))
  CC := kgcc gcc cc
else
  CC := gcc cc
endif
test_cc = $(shell $(cc) --version > /dev/null 2>&1 && echo $(cc))
CC := $(foreach cc, $(CC), $(test_cc))
CC := $(firstword $(CC))
ifeq (,$(CC))
  $(error Compiler not found)
endif

# we need to know what platform the driver is being built on
# some additional features are only built on Intel platforms
ARCH := $(shell uname -m | sed 's/i.86/i386/')
ifeq ($(ARCH),alpha)
  EXTRA_CFLAGS += -ffixed-8 -mno-fp-regs
endif
ifeq ($(ARCH),x86_64)
  EXTRA_CFLAGS += -mcmodel=kernel -mno-red-zone
endif
ifeq ($(ARCH),ppc)
  EXTRA_CFLAGS += -msoft-float
endif
ifeq ($(ARCH),ppc64)
  EXTRA_CFLAGS += -m64 -msoft-float
  LDFLAGS += -melf64ppc
endif

# get the kernel version - we use this to find the correct install path
KVER := $(shell $(CC) $(EXTRA_CFLAGS) -E -dM $(VERSION_FILE) | grep UTS_RELEASE | \
        awk '{ print $$3 }' | sed 's/\"//g')

# assume source symlink is the same as build, otherwise adjust KOBJ
ifneq (,$(wildcard /lib/modules/$(KVER)/build))
ifneq ($(KSRC),$(shell readlink /lib/modules/$(KVER)/build))
  KOBJ=/lib/modules/$(KVER)/build
endif
endif

KKVER := $(shell echo $(KVER) | \
         awk '{ if ($$0 ~ /2\.[4-9]\./) print "1"; else print "0"}')
ifeq ($(KKVER), 0)
  $(error *** Aborting the build. \
          *** This driver is not supported on kernel versions older than 2.4.0)
endif


# set the install path
INSTDIR := /lib/modules/$(KVER)/kernel/drivers/net/$(DRIVER_NAME)

# look for SMP in config.h
SMP := $(shell $(CC) $(EXTRA_CFLAGS) -E -dM $(CONFIG_FILE) | \
         grep -w CONFIG_SMP | awk '{ print $$3 }')
ifneq ($(SMP),1)
  SMP := 0
endif

ifneq ($(SMP),$(shell uname -a | grep SMP > /dev/null 2>&1 && echo 1 || echo 0))
  $(warning ***)
  ifeq ($(SMP),1)
    $(warning *** Warning: kernel source configuration (SMP))
    $(warning *** does not match running kernel (UP))
  else
    $(warning *** Warning: kernel source configuration (UP))
    $(warning *** does not match running kernel (SMP))
  endif
  $(warning *** Continuing with build,)
  $(warning *** resulting driver may not be what you want)
  $(warning ***)
endif

ifeq ($(SMP),1)
  EXTRA_CFLAGS += -D__SMP__
endif

###########################################################################
# 2.4.x & 2.6.x Specific rules

K_VERSION:=$(shell echo $(BUILD_KERNEL) | cut -c1-3 | sed 's/2\.[56]/2\.6/')

ifeq ($(K_VERSION), 2.6)

# Makefile for 2.6.x kernel
TARGET = $(DRIVER_NAME).ko

# man page
MANSECTION = 7
MANFILE = $(TARGET:.ko=.$(MANSECTION))

ifneq ($(PATCHLEVEL),)
EXTRA_CFLAGS += $(CFLAGS_EXTRA)
obj-m += $(DRIVER_NAME).o
$(DRIVER_NAME)-objs := $(CFILES:.c=.o)
else
default:
ifeq ($(KOBJ),$(KSRC))
	$(MAKE) -C $(KSRC) SUBDIRS=$(shell pwd) modules
else
	$(MAKE) -C $(KSRC) O=$(KOBJ) SUBDIRS=$(shell pwd) modules
endif
endif

else # ifeq ($(K_VERSION),2.6)

# Makefile for 2.4.x kernel
TARGET = $(DRIVER_NAME).o

# man page
MANSECTION = 7
MANFILE = $(TARGET:.o=.$(MANSECTION))

# Get rid of compile warnings in kernel header files from SuSE
ifneq (,$(wildcard /etc/SuSE-release))
  EXTRA_CFLAGS += -Wno-sign-compare -fno-strict-aliasing
endif

# Get rid of compile warnings in kernel header files from fedora
ifneq (,$(wildcard /etc/fedora-release))
  EXTRA_CFLAGS += -fno-strict-aliasing
endif

CFLAGS += $(EXTRA_CFLAGS)

.SILENT: $(TARGET)
$(TARGET): $(filter-out $(TARGET), $(CFILES:.c=.o))
	$(LD) $(LDFLAGS) -r $^ -o $@
	echo; echo
	echo "**************************************************"
	echo "** $(TARGET) built for $(KVER)"
	echo -n "** SMP               "
	if [ "$(SMP)" = "1" ]; \
		then echo "Enabled"; else echo "Disabled"; fi
	echo "**************************************************"
	echo

$(CFILES:.c=.o): $(HFILES) Makefile
default:
	$(MAKE)

endif # ifeq ($(K_VERSION),2.6)

ifeq (,$(MANDIR))
  # find the best place to install the man page
  MANPATH := $(shell (manpath 2>/dev/null || echo $MANPATH) | sed 's/:/ /g')
  ifneq (,$(MANPATH))
    # test based on inclusion in MANPATH
    test_dir = $(findstring $(dir), $(MANPATH))
  else
    # no MANPATH, test based on directory existence
    test_dir = $(shell [ -e $(dir) ] && echo $(dir))
  endif
  # our preferred install path
  # should /usr/local/man be in here ?
  MANDIR := /usr/share/man /usr/man
  MANDIR := $(foreach dir, $(MANDIR), $(test_dir))
  MANDIR := $(firstword $(MANDIR))
endif
ifeq (,$(MANDIR))
  # fallback to /usr/man
  MANDIR := /usr/man
endif

# depmod version for rpm builds
DEPVER := $(shell /sbin/depmod -V 2>/dev/null | \
          awk 'BEGIN {FS="."} NR==1 {print $$2}')

###########################################################################
# Build rules

#$(MANFILE).gz: ../$(MANFILE)
#	gzip -c $< > $@

install: #default $(MANFILE).gz
	# remove all old versions of the module
	find $(INSTALL_MOD_PATH)/lib/modules/$(KVER) -name $(TARGET) -exec rm -f {} \; || true
	find $(INSTALL_MOD_PATH)/lib/modules/$(KVER) -name $(TARGET).gz -exec rm -f {} \; || true
	install -D -m 644 $(TARGET) $(INSTALL_MOD_PATH)$(INSTDIR)/$(TARGET)
ifeq (,$(INSTALL_MOD_PATH))
	/sbin/depmod -a || true
else
  ifeq ($(DEPVER),1 )
	/sbin/depmod -r $(INSTALL_MOD_PATH) -a || true
  else
	/sbin/depmod -b $(INSTALL_MOD_PATH) -a -n $(KVERSION) > /dev/null || true
  endif
endif
#	install -D -m 644 $(MANFILE).gz $(INSTALL_MOD_PATH)$(MANDIR)/man$(MANSECTION)/$(MANFILE).gz
#	man -c -P'cat > /dev/null' $(MANFILE:.$(MANSECTION)=) || true

uninstall:
	if [ -e $(INSTDIR)/$(TARGET) ] ; then \
	    rm -f $(INSTDIR)/$(TARGET) ; \
	fi
	/sbin/depmod -a
	if [ -e $(MANDIR)/man$(MANSECTION)/$(MANFILE).gz ] ; then \
		rm -f $(MANDIR)/man$(MANSECTION)/$(MANFILE).gz ; \
	fi

.PHONY: clean install

clean:
ifeq ($(KOBJ),$(KSRC))
	$(MAKE) -C $(KSRC) SUBDIRS=$(shell pwd) clean
else
	$(MAKE) -C $(KSRC) O=$(KOBJ) SUBDIRS=$(shell pwd) clean
endif
	rm -rf $(TARGET) $(TARGET:.ko=.o) $(TARGET:.ko=.mod.c) $(TARGET:.ko=.mod.o) $(CFILES:.c=.o) $(MANFILE).gz .*cmd .tmp_versions /bin/linkctrl
	# remove all old versions of the module
	find $(INSTALL_MOD_PATH)/lib/modules/$(KVER) -name $(TARGET) -exec rm -f {} \; || true
	find $(INSTALL_MOD_PATH)/lib/modules/$(KVER) -name $(TARGET).gz -exec rm -f {} \; || true
