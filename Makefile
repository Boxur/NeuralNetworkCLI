# Makefile for repo C

TARGET_EXEC := neural_network_cli

BUILD_DIR := ./build

# --- Source directories ---
NN_DIR := ./nn
CLI_DIR := ./cli
C_DIR  := ./src
C_INC_DIR := ./include

# --- Sources --
NN_SRCS := $(filter-out $(NN_DIR)/src/neural_network.cpp,$(shell find $(NN_DIR)/src -name '*.cpp'))
CLI_SRCS := $(filter-out $(CLI_DIR)/src/commands.cpp,$(shell find $(CLI_DIR)/src -name '*.cpp'))
C_SRCS   := $(shell find $(C_DIR) -name '*.cpp' -or -name '*.c')
SRCS     := $(NN_SRCS) $(CLI_SRCS) $(C_SRCS)

# --- Include directories ---
NN_INC_DIRS  := $(shell find $(NN_DIR)/include -type d)
CLI_INC_DIRS := $(shell find $(CLI_DIR)/include -type d)
C_INC_DIRS   := $(shell find $(C_INC_DIR) -type d)
INC_FLAGS    := $(addprefix -I,$(NN_INC_DIRS) $(CLI_INC_DIRS) $(C_INC_DIRS))

OBJS := $(SRCS:%=$(BUILD_DIR)/%.o)

# --- Final build ---
$(TARGET_EXEC): $(OBJS)
	$(CXX) -g $(OBJS) -O3 -DNDEBUG -o $@ $(LDFLAGS)

# --- Compile objects ---
$(BUILD_DIR)/%.o: %
	mkdir -p $(dir $@)
ifeq ($(shell echo $< | grep -q '^$(NN_DIR)'; echo $$?),0)
	# NeuralNetwork: force optimization
	$(CXX) -O3 -DNDEBUG $(INC_FLAGS) $(CXXFLAGS) -c $< -o $@
else
	# Commands or C project: normal compile
	$(CXX) $(INC_FLAGS) $(CXXFLAGS) -c $< -o $@
endif

# --- AddressSanitizer build ---
.PHONY: asan
asan: clean $(OBJS)
	@echo "Building with AddressSanitizer..."
	$(CXX) -fsanitize=address -g $(OBJS) -o $(TARGET_EXEC) $(LDFLAGS)

# --- Clean ---
.PHONY: clean
clean:
	rm -rf $(BUILD_DIR)
	rm -f $(TARGET_EXEC)
	rm -f out.perf-folded flamegraph.svg perf.data

# --- Submodule update ---
.PHONY: update-submodules
update-submodules:
	git submodule update --init --recursive
	git submodule foreach git pull origin main

# --- Perf timing ---
.PHONY: time
time: $(TARGET_EXEC)
	perf record -F 99 -g ./$(TARGET_EXEC)
	perf report -n --stdio
