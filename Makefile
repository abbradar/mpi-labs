labs := lab1 lab2 lab3 lab4
common_objs := utils.o
rootdir := $(shell pwd)

.PHONY: clean

define goal_template =
$(eval include $(1)/build.mk)
$(1)/$(1): $(foreach obj,$($(1)_objs),$(1)/$(obj)) $(common_objs)
	mpicc $$^ -o $$@
endef

$(foreach lab, $(labs), \
	$(eval $(call goal_template,$(lab))) \
)

%.o: %.c
	mpicc -std=c99 -I $(rootdir) -c $< -o $@

clean:
	find -name \*.o -delete
	rm -rf $(foreach lab,$(labs),$(lab)/$(lab))
