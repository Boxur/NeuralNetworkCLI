
# ----------------------------
# Makefile for repo C
# ----------------------------

TARGET_EXEC := neural_network_cli
BUILD_DIR   := ./build

# --- Project directories ---
NN_DIR  := nn
CLI_DIR := cli
C_DIR   := src
C_INC_DIR := include

# --- Gather sources (exclude submodule main files) ---
NN_SRCS  := $(filter-out $(NN_DIR)/src/neural_network.cpp,$(shell find $(NN_DIR)/src -name '*.cpp' -or -name '*.c'))
CLI_SRCS := $(filter-out $(CLI_DIR)/src/commands.cpp,$(shell find $(CLI_DIR)/src -name '*.cpp' -or -name '*.c'))
C_SRCS   := $(shell find $(C_DIR) -name '*.cpp' -or -name '*.c')

# --- Include directories ---
NN_INC_DIRS  := $(shell find $(NN_DIR)/include -type d)
CLI_INC_DIRS := $(shell find $(CLI_DIR)/include -type d)
C_INC_DIRS   := $(shell find $(C_INC_DIR) -type d)
INC_FLAGS    := $(addprefix -I,$(NN_INC_DIRS) $(CLI_INC_DIRS) $(C_INC_DIRS))

# --- Objects ---
OBJS := $(NN_SRCS:$(NN_DIR)/%.cpp=$(BUILD_DIR)/$(NN_DIR)/%.o) \
        $(CLI_SRCS:$(CLI_DIR)/%.cpp=$(BUILD_DIR)/$(CLI_DIR)/%.o) \
        $(C_SRCS:$(C_DIR)/%.cpp=$(BUILD_DIR)/$(C_DIR)/%.o)

CXXFLAGS := 

# ----------------------------
# Final build
# ----------------------------
$(TARGET_EXEC): $(OBJS)
	$(CXX) -g $(OBJS) -O3 -DNDEBUG -o $@ $(LDFLAGS)

# ----------------------------
# Compile rules for each project
# ----------------------------

# NeuralNetwork (optimized)
$(BUILD_DIR)/nn/%.o: nn/%.cpp
	mkdir -p $(dir $@)
	$(CXX) -O3 -DNDEBUG $(INC_FLAGS) $(CXXFLAGS) -c $< -o $@

# Commands (normal)
$(BUILD_DIR)/cli/%.o: cli/%.cpp
	mkdir -p $(dir $@)
	$(CXX) $(INC_FLAGS) $(CXXFLAGS) -c $< -o $@

# Main C project (normal)
$(BUILD_DIR)/src/%.o: src/%.cpp
	mkdir -p $(dir $@)
	$(CXX) $(INC_FLAGS) $(CXXFLAGS) -c $< -o $@

# ----------------------------
# AddressSanitizer build
# ----------------------------
.PHONY: asan
asan: clean $(OBJS)
	@echo "Building with AddressSanitizer..."
	$(CXX) -fsanitize=address -g $(OBJS) -o $(TARGET_EXEC) $(LDFLAGS)

# ----------------------------
# Clean
# ----------------------------
.PHONY: clean
clean:
	rm -rf $(BUILD_DIR)
	rm -f $(TARGET_EXEC)
	rm -f out.perf-folded flamegraph.svg perf.data

# ----------------------------
# Submodule update
# ----------------------------
.PHONY: update-submodules
update-submodules:
	git submodule update --init --recursive
	git submodule foreach git fetch origin
	git submodule foreach git reset --hard origin/main

# ----------------------------
# Perf timing
# ----------------------------
.PHONY: time
time: $(TARGET_EXEC)
	perf record -F 99 -g ./$(TARGET_EXEC)
	perf report -n --stdio
