NVCC := nvcc

NVCC_FLAGS := -O3 -std=c++14 -I./src

# 3. 目标可执行文件名
TARGET := benchmark_reduce

# 4. 源文件和目标对象
# 这里会自动找到当前目录下的 main.cu
SRCS := main.cu
OBJS := $(SRCS:.cu=.o)

# ==========================================
# 编译规则 (Rules)
# ==========================================

# 默认目标，当你输入 `make` 时执行
all: $(TARGET) run

# 链接生成最终的可执行文件
$(TARGET): $(OBJS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^
	@echo "Build complete! Run with ./$(TARGET)"

# 编译 .cu 文件生成 .o 目标文件
%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# 清理编译产物，当你输入 `make clean` 时执行
clean:
	rm -f $(OBJS) $(TARGET)
	@echo "Clean complete!"

run : $(TARGET)
	@echo "Running benchmark..."
	./$(TARGET)

# 声明伪目标，防止和同名文件冲突
.PHONY: all clean