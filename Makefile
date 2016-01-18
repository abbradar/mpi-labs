MPICC ?= mpicc
CFLAGS = -Wall -std=c99 -O -g -DOMPI -DOMP -fopenmp
LDFLAGS = -lm

labs := lab1 lab2 lab3 lab4 lab5 task
common_objs := utils.o

rootdir ?= $(shell pwd)
installdir ?= /gpfs/home/iu7/$(shell whoami)/
nodes ?= 8

.PHONY: clean cleaninstall $(foreach lab,$(labs),$(lab).run)

define goal_template
$(eval include $(1)/build.mk)
# FIXME: maybe this can be done better?
$(eval $(1)_nodes ?= $(nodes))

$(1)/$(1): $(foreach obj,$($(1)_objs),$(1)/$(obj)) $(common_objs)
	$(MPICC) $(CFLAGS) $(LDFLAGS) $$^ -o $$@

# Sets installdir in the job file, among other things
$(1)/$(1).job: lab.job.in
	sed \
	  -e 's,@job_name@,$(1),g' \
	  -e 's,@initialdir@,$(installdir),' \
	  -e 's,@node@,$($(1)_nodes),' \
	  $$< > $$@

# Copy executable to installdir and run the job
$(1).run: $(1)/$(1) $(1)/$(1).job
	install -m755 $(1)/$(1) $(installdir)/$(1)
	llsubmit $(1)/$(1).job
endef

$(foreach lab, $(labs), \
	$(eval $(call goal_template,$(lab))) \
)

%.o: %.c
	$(MPICC) $(CFLAGS) -I $(rootdir) -c $< -o $@

clean:
	find -name \*.o -delete
	rm -rf $(foreach lab,$(labs),$(lab)/$(lab) $(lab)/$(lab).job)

cleaninstall:
	rm -rf $(foreach lab,$(labs),$(installdir)/$(lab) $(installdir)/$(lab).*.{stdout,stderr} $(installdir)/$(lab).trace* $(installdir)/$(installdir)/$(lab).ttf)
