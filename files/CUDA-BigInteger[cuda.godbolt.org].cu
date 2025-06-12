#include <cuda_runtime.h>
#include <cufft.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cub/cub.cuh>
#include <cub/util_allocator.cuh>
#include <iostream>
#include <vector>

#include <stdio.h>
#include <math.h>


#include <iostream>
#include <cmath>  // std::round

#include <curand_kernel.h>//    // 初始化亂數種子
    //curandState state;
    //curand_init(1234, idx, 0, &state);


using namespace std;

class AllocatorManager {
   public:
    static cub::CachingDeviceAllocator& GetInstance() {
        static cub::CachingDeviceAllocator instance;
        return instance;
    }
};

#define CUDA_CALL(func)                                                       \
    {                                                                         \
        cudaError_t err = (func);                                             \
        if (err != cudaSuccess) {                                             \
            std::cerr << "CUDA error in " << #func << " at line " << __LINE__ \
                      << ": " << cudaGetErrorString(err) << std::endl;        \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    }

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <vector>

using namespace std;


// basebits存儲log2(base)而不是直接的base值；然後使用時long long base = 1LL << 32;
const int basebits = 3;  // uint32_t
const long long base = 1LL << basebits;




__global__ void bigIntegerMultiplyKernel(const int* a, int aSize, const int* b,
                                         int bSize, int* result) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= aSize * bSize) return;

    int i = idx / bSize;
    int j = idx % bSize;

    unsigned long long product =
        (unsigned long long)a[i] * (unsigned long long)b[j];
    int low = product % base;
    int high = product / base;

    atomicAdd(&result[i + j], low);
    atomicAdd(&result[i + j + 1], high);
}

class BigInteger {
   public:
    int sign;  // sign = -1, Negative; sign = 0, Zero; sign = +1, Positive;
    int size;
    int* digits;  // 每位存储一个数字（低位到高位）digits[0] = d1; for
                  // BigInteger [dn----d3d2d1] = d1 + d2*base + d3*base^2 + ...

    __host__ __device__ BigInteger() : sign(0), size(0), digits(nullptr) {}
    __host__ __device__ BigInteger(long long n) : sign(0), size(0), digits(nullptr) {
        if (n == 0) {
            sign = 0;
            size = 0;
            digits = nullptr;
        } else {
            // Determine sign
            if (n > 0) {
                sign = +1;
            } else {
                sign = -1;
                n = -n;  // Make n positive for the rest of the calculation
            }

            // Compute the size based on base
            int tmp = n;
            size = 0;
            while (tmp > 0) {
                tmp /= base;
                size++;
            }

            // Allocate memory for digits
            digits = new int[size];

            // Fill digits array
            for (int i = 0; i < size; ++i) {
                digits[i] = n % base;
                n /= base;
            }
        }
    }

    // 构造函数接管 digits 的所有权，并在析构函数中释放
    // Constructor from sign and vector of digits
    __host__ __device__ BigInteger(int sgn, int* digits, int size) : sign(sgn), size(size) {
        this->digits = new int[size];
        for (int i = 0; i < size; ++i) {
            this->digits[i] = digits[i];
        }
    }


    // 复制构造函数（深拷贝）
    __host__ __device__ BigInteger(const BigInteger& other) {
        sign = other.sign;
        size = other.size;

        if (other.digits != nullptr) {
            digits = new int[size];  // Allocate memory for digits
            for (int i = 0; i < size; ++i) {
                digits[i] = other.digits[i];
            }
        } else {
            digits = nullptr;
        }
    }

    static __device__ BigInteger scaleUpTo(const double d, const int logp) {
        BigInteger result(0);
        double cnst = d;
        if (d < 0) cnst = -d;

        long long IntegerPart = (long long)cnst;
        double FractionalPart = cnst - IntegerPart;

        BigInteger integer(IntegerPart);
        result = result + integer << logp;

        long long two60 = 1LL;
        if(logp < 60) {
            two60 = 1LL << logp;
            BigInteger fractional(  std::round(FractionalPart*two60)  );
            result = result + fractional;
        } else {
            two60 = 1LL << 60;
            BigInteger fractional(  std::round(FractionalPart*two60)  );
            fractional = fractional << (logp - 60);
            result = result + fractional;            
        }

        if (d < 0) result.sign = -1;

        return result;

    }

    // 赋值操作符（深拷贝）
    __host__ __device__ BigInteger& operator=(const BigInteger& other) {
        if (this == &other) return *this;  // Self-assignment check

        sign = other.sign;
        size = other.size;

        if (digits != nullptr) {
            delete[] digits;  // Free existing memory
        }

        if (other.digits != nullptr) {
            digits = new int[size];  // Allocate new memory
            for (int i = 0; i < size; ++i) {
                digits[i] = other.digits[i];
            }
        } else {
            digits = nullptr;
        }

        return *this;
    }

    __device__ void trimLeadingZeros() {
        while (size > 0 && digits[size - 1] == 0) {
            size--;
        }
        if (size == 0) {
            sign = 0;  // 0 是非负数
            if (digits) {
                delete[] digits;   // 释放已分配的内存
                digits = nullptr;  // 避免野指针
            }
        }
    }

    // 比较绝对值
    __device__ int compareMagnitude(const BigInteger* a,
                                    const BigInteger* b) const {
        int maxsize = a->size;
        if (a->size < b->size) {
            maxsize = b->size;
        }

        for (int i = maxsize - 1; i >= 0; --i) {
            int digitA = (i < a->size) ? a->digits[i] : 0;
            int digitB = (i < b->size) ? b->digits[i] : 0;

            if (digitA != digitB) {
                return (digitA < digitB) ? -1 : 1;
            }
        }

        return 0;  // 相等
    }

    __device__ bool operator==(const BigInteger& b) const {
        if (sign != b.sign) return false;

        const BigInteger a = *this;
        int cmp = compareMagnitude(&a, &b);
        if (cmp == 0)
            return true;
        else
            return false;
    }
    __device__ bool operator!=(const BigInteger& other) const {
        return !(*this == other);
    }
    __device__ bool operator<(const BigInteger& b) const {
        // 先比较符号
        if (sign != b.sign) return sign < b.sign;

        const BigInteger a = *this;
        int cmp = compareMagnitude(&a, &b);
        if (a.sign == -1 && cmp == +1) return true;
        if (a.sign == +1 && cmp == -1) return true;

        return false;  // 相等的情况
    }
    __device__ bool operator>(const BigInteger& other) const {
        return other < *this;
    }
    __device__ bool operator<=(const BigInteger& other) const {
        return !(other < *this);
    }
    __device__ bool operator>=(const BigInteger& other) const {
        return !(*this < other);
    }

    __device__ BigInteger operator-() const {
        // 創建一個新的 BigInteger 用來表示取逆的結果
        BigInteger result(*this);    // 使用拷貝構造函數
        result.sign = -result.sign;  // 反轉符號
        return result;
    }
    __device__ BigInteger& operator-() {

        sign = -sign;

        return *this;
    }


    // 计算大整数的二进制位数
    __device__ int bitLength() const {
        if (sign == 0 || digits == nullptr || size == 0) {
            return 1;  // 空或未初始化的数字 1111111111
        }

        int bitsnum = int(log2f(base));
        //printf("\nbitsnum:%d\n", bitsnum);

        int totalBits = 0;
        bool flag = false;
        // 从高位开始，遇到第一个非零的digits[i]，之后的所有数字（包括零）都直接算作 base 的位数
        for (int i = size - 1; i >= 0; --i) {
            if (digits[i] != 0 && !flag) {
                // 遇到第一个非零数字，开始计算后续数字的位数
                totalBits += (int)(log2f(digits[i])) + 1;  // 计算当前“数字”的二进制位数
                flag = true;                     // 标记已经开始计算
            } else if (flag) {
                // 如果已经遇到第一个非零数字，后面的数字（不管是0还是非0）都算作
                // base 的位数
                totalBits += bitsnum;
            }
        }
        return totalBits;
    }

    // 確保BigInteger對象的初始化生成時候每一個digit<base  這樣可以>> int
    // maxSize = 1;  // 为结果预留空间
    //  绝对值加法
    __device__ BigInteger* addMagnitude(const BigInteger* b) const {
        int maxSize = 1;  // 为结果预留空间
        if (size < b->size)
            maxSize += b->size;
        else
            maxSize += size;

        BigInteger* result = (BigInteger*)malloc(
            sizeof(BigInteger));  // 在设备内存中动态分配内存
        result->digits =
            (int*)malloc(maxSize * sizeof(int));  // 为 digits 分配内存
        result->size = maxSize;

        int carry = 0;
        for (int i = 0; i < result->size; ++i) {
            int digitA = (i < size) ? digits[i] : 0;
            int digitB = (i < b->size) ? b->digits[i] : 0;
            unsigned long long sum =
                (unsigned long long)digitA + (unsigned long long)digitB + carry;
            result->digits[i] = int(sum % base);
            carry = sum / base;
        }

        // 处理进位
        // 去除前导零
        result->trimLeadingZeros();

        return result;
    }

    // 绝对值减法
    __device__ BigInteger* subtractMagnitude(const BigInteger* b) const {
        int maxSize = size;  // 结果的最大大小通常是 a 的大小（b 不会比 a 更大）
        if (maxSize < b->size) maxSize = b->size;

        BigInteger* result = (BigInteger*)malloc(sizeof(BigInteger));
        result->digits = (int*)malloc(maxSize * sizeof(int));
        result->size = maxSize;

        int borrow = 0;
        for (int i = 0; i < maxSize; ++i) {
            int digitA = (i < size) ? digits[i] : 0;
            int digitB = (i < b->size) ? b->digits[i] : 0;
            int diff = digitA - digitB - borrow;

            if (diff < 0) {
                diff += base;
                borrow = 1;
            } else {
                borrow = 0;
            }

            result->digits[i] = diff;
        }

        // 去除前导零
        result->trimLeadingZeros();

        return result;
    }
    // 符号重载（加法）
    __device__ BigInteger operator+(const BigInteger& b) const{
        // 如果符号相同，执行绝对值加法
        if (sign == b.sign) {
            BigInteger* result = addMagnitude(&b);
            result->sign = sign;  // 结果符号和当前对象相同
            return *result;
        }

        // 如果符号不同，执行绝对值减法，符号取较大的绝对值的符号
        int cmp = compareMagnitude(this, &b);
        if (cmp == 0) {
            // 两个数相等，结果为零
            BigInteger* result = (BigInteger*)malloc(sizeof(BigInteger));
            result->sign = 0;
            result->size = 0;
            result->digits = nullptr;
            return *result;
        } else if (cmp > 0) {
            BigInteger* result = subtractMagnitude(&b);
            result->sign = sign;
            return *result;
        } else {
            const BigInteger temp(*this);
            BigInteger* result = b.subtractMagnitude(&temp);
            result->sign = b.sign;
            return *result;
        }
    }
    __device__ BigInteger& operator+=(const BigInteger& b){
        if(digits) free(digits);
        digits = nullptr;

        // 如果符号相同，执行绝对值加法
        if (sign == b.sign) {
            *this = *addMagnitude(&b);
            this->sign = sign;  // 结果符号和当前对象相同
            return *this;
        }

        // 如果符号不同，执行绝对值减法，符号取较大的绝对值的符号
        int cmp = compareMagnitude(this, &b);
        if (cmp == 0) {
            // 两个数相等，结果为零
            this->sign = 0;
            this->size = 0;
            this->digits = nullptr;
            return *this;
        } else if (cmp > 0) {
            *this = *subtractMagnitude(&b);
            this->sign = sign;
            return *this;
        } else {
            const BigInteger temp(*this);
            *this = *b.subtractMagnitude(&temp);
            this->sign = b.sign;
            return *this;
        }
    }

    // 符号重载（减法）
    __device__ BigInteger operator-(const BigInteger& b) const {
        // 如果符号相同，执行绝对值减法，符号取较大的绝对值的符号
        if (sign == b.sign) {
            int cmp = compareMagnitude(this, &b);
            if (cmp == 0) {
                // 两个数相等，结果为零
                BigInteger* result = (BigInteger*)malloc(sizeof(BigInteger));
                result->sign = 0;
                result->size = 0;
                result->digits = nullptr;
                return *result;
            } else if (cmp > 0) {
                BigInteger* result = subtractMagnitude(&b);
                result->sign = sign;
                return *result;
            } else {
                const BigInteger temp(*this);
                BigInteger* result = b.subtractMagnitude(&temp);
                result->sign = -b.sign;
                return *result;
            }
        }

        // 如果符号不同，执行加法
        BigInteger* result = addMagnitude(&b);
        result->sign = sign;  // 保持当前符号
        return *result;
    }
    __device__ BigInteger& operator-=(const BigInteger& b) {
        if(digits) free(digits);
        digits = nullptr;

        // 如果符号相同，执行绝对值减法，符号取较大的绝对值的符号
        if (sign == b.sign) {
            int cmp = compareMagnitude(this, &b);
            if (cmp == 0) {
                // 两个数相等，结果为零
                this->sign = 0;
                this->size = 0;
                this->digits = nullptr;
                return *this;
            } else if (cmp > 0) {
                *this = *subtractMagnitude(&b);
                this->sign = sign;
                return *this;
            } else {
                const BigInteger temp(*this);
                *this = *b.subtractMagnitude(&temp);
                this->sign = -b.sign;
                return *this;
            }
        }

        // 如果符号不同，执行加法
        *this = *addMagnitude(&b);
        this->sign = sign;  // 保持当前符号
        return *this;
    }

    // 可以在核函數中使用FFT via cuFFTDx來加速大整數的乘法計算
    // 不過要依賴庫並且需要CUDA版本大於11.0
    __device__ BigInteger operator*(const BigInteger& b) const {     
        // if sgn == 0 or other.sgn == 0
        //if (sgn == 0 || other.sgn == 0) return BigInteger("0");

        // if digits.size()==1 && digits[0] == 1: 
        //      sgn == +1 or -1?
        // if other.digits.size()==1 && other.digits[0] == 1: 
        //      sgn == +1 or -1?
        // if this or other is a power of two: just << or >>


   
        int resultSize = size + b.size + 3;

        BigInteger* result = (BigInteger*)malloc(sizeof(BigInteger));
        result->digits = (int*)malloc(resultSize * sizeof(int));
        result->size = resultSize;
        result->sign = sign * b.sign;
        for (int i = 0; i < resultSize; ++i) result->digits[i] = 0;

        // 通过两个for循环执行乘法
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < b.size; ++j) {
                unsigned long long product = (unsigned long long)digits[i] * (unsigned long long)b.digits[j];
                int low = product % base;  // 低位部分
                int high = product / base;  // 高位部分

                result->digits[i + j] += low;  // 累加低位
                result->digits[i + j + 1] += high;  // 累加高位
            }
        }


        // 处理进位
        unsigned long long carry = 0;
        for (size_t i = 0; i < result->size; ++i) {
            unsigned long long sum = (unsigned long long )result->digits[i] + carry;
            result->digits[i] = (int)(sum % base);
            carry = sum / base;
        }

        result->trimLeadingZeros();
        return *result;
    }
    __device__ BigInteger& operator*=(const BigInteger& b) {     
        if(digits) free(digits);
        digits = nullptr;
        // if sgn == 0 or other.sgn == 0
        //if (sgn == 0 || other.sgn == 0) return BigInteger("0");

        // if digits.size()==1 && digits[0] == 1: 
        //      sgn == +1 or -1?
        // if other.digits.size()==1 && other.digits[0] == 1: 
        //      sgn == +1 or -1?
        // if this or other is a power of two: just << or >>


   
        int resultSize = size + b.size + 3;

        //this = (BigInteger*)malloc(sizeof(BigInteger));
        this->digits = (int*)malloc(resultSize * sizeof(int));
        this->size = resultSize;
        this->sign = sign * b.sign;
        for (int i = 0; i < resultSize; ++i) this->digits[i] = 0;

        // 通过两个for循环执行乘法
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < b.size; ++j) {
                unsigned long long product = (unsigned long long)digits[i] * (unsigned long long)b.digits[j];
                int low = product % base;  // 低位部分
                int high = product / base;  // 高位部分

                this->digits[i + j] += low;  // 累加低位
                this->digits[i + j + 1] += high;  // 累加高位
            }
        }


        // 处理进位
        unsigned long long carry = 0;
        for (size_t i = 0; i < this->size; ++i) {
            unsigned long long sum = (unsigned long long )this->digits[i] + carry;
            this->digits[i] = (int)(sum % base);
            carry = sum / base;
        }

        this->trimLeadingZeros();
        return *this;
    }
 

    // for debugging .........
    // 从 BigInteger 返回 long long 类型
    __device__ long long toLongLong() const {
        long long result = 0;
        long long factor = 1;
        for (int i = 0; i < size; ++i) {
            result += digits[i] * factor;
            factor *= base;
        }

        return result * sign;
    }

    // Left shift operation (multiply by 2^count)
    __device__ BigInteger operator<<(int count) const {
        // //assert(count >= 0);

        int bitsnum = int(log2f(base));
        //printf("\nbitsnum:%d\n", bitsnum);

        int wordcount = count / bitsnum;
        int bitcount = count % bitsnum;

        BigInteger* result = (BigInteger*)malloc(sizeof(BigInteger));
        result->sign = sign;
        result->size = size + wordcount + 1;
        result->digits = (int*)malloc(result->size * sizeof(int));

        for (int i = 0; i < wordcount; ++i) result->digits[i] = 0;

        int carry = 0;
        for (int i = 0; i < size; ++i) {
            auto shifted = (((unsigned long long)digits[i]) << bitcount);
            result->digits[wordcount + i] = int((shifted + carry) % base);
            carry = int((shifted + carry) / base);
        }
        result->digits[wordcount + size] = carry;

        result->trimLeadingZeros();

        return *result;
    }
    __device__ BigInteger& operator<<=(int count) {
        if(digits) free(digits);
        digits = nullptr;

        // //assert(count >= 0);

        int bitsnum = int(log2f(base));
        //printf("\nbitsnum:%d\n", bitsnum);

        int wordcount = count / bitsnum;
        int bitcount = count % bitsnum;

        //this = (BigInteger*)malloc(sizeof(BigInteger));
        this->sign = sign;
        this->size = size + wordcount + 1;
        this->digits = (int*)malloc(this->size * sizeof(int));

        for (int i = 0; i < wordcount; ++i) this->digits[i] = 0;

        int carry = 0;
        for (int i = 0; i < size; ++i) {
            auto shifted = (((unsigned long long)digits[i]) << bitcount);
            this->digits[wordcount + i] = int((shifted + carry) % base);
            carry = int((shifted + carry) / base);
        }
        this->digits[wordcount + size] = carry;

        this->trimLeadingZeros();

        return *this;
    }

    // C++效果：x >> n 表示將 x 向右移動 n 位，空出來的位根據類型填充：
    // 無符號整數 (unsigned int、uint64_t)：高位補 0（邏輯右移）。
    // 有符號整數 (int、long、long long)：
    // 大多數情況（GCC, Clang, MSVC）：使用算術右移（即高位補符號位 0 或 1）。
    // 特殊情況：標準沒有保證一定是算術右移（某些編譯器可能會實現為邏輯右移）。
    //-35 >> 2 == -9
    // NTL::ZZ中所使用的是不考慮補碼 而Python3和Java中的採用的是等價於
    // BigInteger 除以 2^n，並且 對於負數，仍然會進行算術右移（保留符號）。 這與
    // Java 基本型別的 >> 行為一致。算術右移，負數補 1，正數補 0 在 Java
    // 中，對於 基本型別的整數（int、long），>>（右位移）與 Python 類似，即
    // 將位元向右移動並保留符號位（負數補 1，正數補 0），這稱為 算術右移。 但
    // 對於 BigInteger（大整數），Java 提供了一個方法 shiftRight(int n)
    // 來執行右位移操作，這與 >> 作用相同。
    // ------------
    // 在同態加密庫HEAAN中可能並沒有太大作用應該沒用到這個操作？？？
    // ------------ Right shift operation (divide by 2^count)
    __device__ BigInteger operator>>(int count) const {
        //assert(count >= 0);

        int bitsnum = int(log2f(base));
        //printf("\nbitsnum:%d\n", bitsnum);

        int wordcount = count / bitsnum;
        int bitcount = count % bitsnum;

        // printf("\ncount%d  \n", count);
        // printf("\nbitsnum%d  \n", bitsnum);
        // printf("\nwordcount%d  \n", wordcount);
        // printf("\bitcount%d  \n", bitcount);
        // printf("size%d  \n", size);

        BigInteger* result = (BigInteger*)malloc(sizeof(BigInteger));
        result->sign = sign;

        if (sign == 0) {  // printf("\bitcount%d  \n", bitcount);
            result->sign = 0;
            result->size = 0;
            result->digits = nullptr;

            return *result;
        }

        if (wordcount >= size) {  // printf("\nsize%d  \n", size);
            // if(sign == +1) {
            result->sign = 0;
            result->size = 0;
            result->digits = nullptr;

            return *result;
            // }
            // if(sign == -1) {
            //     result->sign = -1;
            //     result->size = 1;
            //     result->digits = (int*)malloc(result->size * sizeof(int));
            //     result->digits[0] = 1;

            //     return *result;
            // }
        }

        // (wordcount < size) {
        result->size = size - wordcount;
        result->digits = (int*)malloc(result->size * sizeof(int));

        int carry = 0;
        for (int i = size - 1; i >= wordcount; --i) {
            auto shifted = digits[i] >> bitcount;
            result->digits[i - wordcount] = shifted + carry;
            carry = (digits[i] << (bitsnum - bitcount)) & (base - 1);
        }

        result->trimLeadingZeros();
        // if(result->sign == 0 && sign == -1) {
        //     result->sign = -1;
        //     result->size = 1;
        //     result->digits[0] = 1;

        return *result;
    }
    __device__ BigInteger& operator>>=(int count) {
        if(digits) free(digits);
        digits = nullptr;

        //assert(count >= 0);

        int bitsnum = int(log2f(base));
        //printf("\nbitsnum:%d\n", bitsnum);

        int wordcount = count / bitsnum;
        int bitcount = count % bitsnum;

        // printf("\ncount%d  \n", count);
        // printf("\nbitsnum%d  \n", bitsnum);
        // printf("\nwordcount%d  \n", wordcount);
        // printf("\bitcount%d  \n", bitcount);
        // printf("size%d  \n", size);

        //this = (BigInteger*)malloc(sizeof(BigInteger));
        this->sign = sign;

        if (sign == 0) {  // printf("\bitcount%d  \n", bitcount);
            this->sign = 0;
            this->size = 0;
            this->digits = nullptr;

            return *this;
        }

        if (wordcount >= size) {  // printf("\nsize%d  \n", size);
            // if(sign == +1) {
            this->sign = 0;
            this->size = 0;
            this->digits = nullptr;

            return *this;
            // }
            // if(sign == -1) {
            //     result->sign = -1;
            //     result->size = 1;
            //     result->digits = (int*)malloc(result->size * sizeof(int));
            //     result->digits[0] = 1;

            //     return *result;
            // }
        }

        // (wordcount < size) {
        this->size = size - wordcount;
        this->digits = (int*)malloc(this->size * sizeof(int));

        int carry = 0;
        for (int i = size - 1; i >= wordcount; --i) {
            auto shifted = digits[i] >> bitcount;
            this->digits[i - wordcount] = shifted + carry;
            carry = (digits[i] << (bitsnum - bitcount)) & (base - 1);
        }

        this->trimLeadingZeros();
        // if(result->sign == 0 && sign == -1) {
        //     result->sign = -1;
        //     result->size = 1;
        //     result->digits[0] = 1;

        return *this;
    }

    // 计算模 2^n
    //__device__ BigInteger mod2n(int n) const {
    __device__ BigInteger operator%(int n) const {
        //assert(n >= 0);
        if (n < 0) {
            printf("This modulus operation requires a integer, which means n should not be negative.");
            assert(false);
        }

        BigInteger* result = (BigInteger*)malloc(sizeof(BigInteger));
        result->sign = +1;

        // 如果数字是0，返回0
        if (sign == 0) {
            result->sign = 0;
            result->size = 0;
            result->digits = nullptr;

            return *result;
        }

 // int bitsnum = int(log2f(base));
 //        std::cout <<bitsnum;
        
        int bitsnum = int(log2f(base));

        //printf("\nbitsnum:%d\n", bitsnum);

        int wordcount = n / bitsnum;
        int bitcount = n % bitsnum;

        int maxSize = wordcount;
        if (bitcount > 0) maxSize += 1;
        result->size = maxSize;
        result->digits = (int*)malloc(result->size * sizeof(int));

        for (int i = 0; i < wordcount; ++i) {
    //result->digits[i] = wordcount < size ? digits[i] : 0;
            result->digits[i] = i < size ? digits[i] : 0;
        }
        if (bitcount > 0 && wordcount < size) {
            int mask = (1LL << bitcount) - 1;  // 生成一个包含 n 个 1 的掩码
            result->digits[wordcount] = digits[wordcount] & mask;
        }

        // 结果是最低n位，取模操作相当于保留数字的最低n位

        BigInteger ONE(1);
        if (sign == -1) *result = (ONE << n) - *result;

        // 返回结果
        return *result;
    }
     __device__ BigInteger& operator%=(int n) {
        if(digits) free(digits);
        digits = nullptr;

        //assert(n >= 0);
        if (n < 0) {
            printf("This modulus operation requires a integer, which means n should not be negative.");
            assert(false);
        }

        //this = (BigInteger*)malloc(sizeof(BigInteger));
        this->sign = +1;

        // 如果数字是0，返回0
        if (sign == 0) {
            this->sign = 0;
            this->size = 0;
            this->digits = nullptr;

            return *this;
        }

 // int bitsnum = int(log2f(base));
 //        std::cout <<bitsnum;
        
        int bitsnum = int(log2f(base));

        //printf("\nbitsnum:%d\n", bitsnum);

        int wordcount = n / bitsnum;
        int bitcount = n % bitsnum;

        int maxSize = wordcount;
        if (bitcount > 0) maxSize += 1;
        this->size = maxSize;
        this->digits = (int*)malloc(this->size * sizeof(int));

        for (int i = 0; i < wordcount; ++i) {
    //result->digits[i] = wordcount < size ? digits[i] : 0;
            this->digits[i] = i < size ? digits[i] : 0;
        }
        if (bitcount > 0 && wordcount < size) {
            int mask = (1LL << bitcount) - 1;  // 生成一个包含 n 个 1 的掩码
            this->digits[wordcount] = digits[wordcount] & mask;
        }

        // 结果是最低n位，取模操作相当于保留数字的最低n位

        BigInteger ONE(1);
        if (sign == -1) *this = (ONE << n) - *this;

        // 返回结果
        return *this;
    }
 
    /*
        你的程序崩溃的原因是 double free（重复释放内存），主要原因可能出现在
BigInteger 的 digits 指针被错误释放两次。🚨

        问题分析
        主机端 h_b 使用 vector<BigInteger>，但 BigInteger::digits 直接指向
h_ints 数据：

        这意味着 BigInteger 只是一个 浅拷贝，它的 digits 指针实际上指向了 h_ints
里面的 vector<int> 的数据。 h_b 的 vector<BigInteger> 会在 main
结束时自动释放，而 h_ints 也会释放。 如果 BigInteger 析构函数尝试
free(digits)，那么就会发生 double free。 设备端 d_b 也分配了 BigInteger，但
digits 仍然指向 d_ints：

        在 allocateDeviceMemory 里，每个 BigInteger 的 digits 被 cudaMemcpy
赋值为 d_ints[i][j]。 如果 BigInteger 的析构函数释放 digits，但 d_ints[i][j]
也被手动释放，会造成二次释放错误。


        1. BigInteger 的析构函数
你已经去掉了 ~BigInteger() 里的 delete[] digits;，这可以避免 digits
被重复释放。但是，如果 BigInteger 的 digits 被 new
申请的内存管理，那么仍然可能会有 内存泄漏。 改进方案：

确保 digits 不是野指针
在 digits 指向外部数据时，避免重复释放

    */
    // __host__ __device__ ~BigInteger() {
    //     // **删除这行以避免 double free**  還要同時刪除~Ring()中的兩行才行
    //     // if (digits) free(digits);
    // }
    // 重新編寫了構造函數 應該需要free(digits); 不知道爲什麼不行還得空操作
    __host__ __device__ ~BigInteger() {
        // if (digits) free(digits);
        // delete[] digits;
    }
    // // 析构函数，确保释放动态分配的内存
    // __host__ __device__ ~BigInteger() {
    //     if (digits != nullptr) {
    //         delete[] digits;
    //     }
    // }
};


__global__ void testBigIntegerOperations() {
  printf("\n\n\n\n\n\n\n\n\n\n$$$$$$$\n\n\n\n\n\n\n\n\n\n");
  printf("in the function: testBigIntegerOperations()\n");

 printf("BigInteger::base = %lld\n\n\n\n", base);


    // 创建两个 BigInteger 对象 a 和 b
    BigInteger a;
    BigInteger b;

    // 初始化 a 和 b 的 digits 数组，注意这里是假设你有一个方法来填充大整数
    a.size = 3;
    a.digits = (int*)malloc(a.size * sizeof(int));
    a.digits[0] = 2;  // 2
    a.digits[1] = 1;  // 10
    a.digits[2] = 3;  // 100
    a.sign = 1;

    b.size = 3;
    b.digits = (int*)malloc(b.size * sizeof(int));
    b.digits[0] = 4;  // 4
    b.digits[1] = 2;  // 20
    b.digits[2] = 5;  // 200
    b.sign = 1;

    // 执行加法操作
    BigInteger resultAdd = a + b;  // 使用符号重载执行加法
    printf("Addition result: ");
    for (int i = 0; i < resultAdd.size; ++i) {
        printf("%d ", resultAdd.digits[i]);
    }
    printf("\n");

    // 执行减法操作
    BigInteger resultSubtract = a - b;  // 使用符号重载执行减法
    printf("Subtraction result: \n");
    printf("Subtraction result: %d\n", resultSubtract.size);
    printf("Subtraction result: %d\n", resultSubtract.sign);
    for (int i = 0; i < resultSubtract.size; ++i) {
        printf("%d ", resultSubtract.digits[i]);
    }
    printf("\nSubtraction\n\n\n\n\n");




    BigInteger scaledDouble = BigInteger::scaleUpTo(1.25, 2); 
    printf("scaleUpTo result: \n");
    printf("scaleUpTo result: %d\n", scaledDouble.size);
    printf("scaleUpTo result: %d\n", scaledDouble.sign);
    for (int i = 0; i < scaledDouble.size; ++i) {
        printf("%d ", scaledDouble.digits[i]);
    }
    printf("\nscaledDouble\n\n\n\n\n");




    // 清理内存
    free(a.digits);
    free(b.digits);

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    // 初始化亂數種子
    curandState state;
    curand_init(1234, idx, 0, &state);
    for (int i = 0; i < 12; ++i) {
        printf("\n\n\n\n\n\n\n\n================================================================");

        int sign_second = 1 - 2 * (curand(&state) % 2);  // 隨機分配符號 (+1 或 -1) 給第二個 Ring
        int size_second = curand(&state) % 1024;  // 隨機大小的 digits (1 ~ 6)
        int num = sign_second * size_second;

        BigInteger bc(num);
        printf("---sign%d\n", bc.sign);
        printf("---size%d\n", bc.size);
        for (int i = 0; i < bc.size; ++i) printf("%d", bc.digits[i]);
        printf("\n");
        printf("%lld", bc.toLongLong());
        printf("\n");
        BigInteger cc = bc << 3;
        printf("\n----------bc << 3;----------\n");
        printf("+++sign%d\n", cc.sign);
        printf("+++size%d\n", cc.size);
        for (int i = 0; i < cc.size; ++i) printf("%d", cc.digits[i]);
        printf("\n");
        printf("%lld", cc.toLongLong());
        printf("\n");
        BigInteger dc = bc >> 3;
        printf("\n----------bc >> 3;----------\n");
        printf("+++sign%d\n", dc.sign);
        printf("+++size%d\n", dc.size);
        for (int i = 0; i < dc.size; ++i) printf("%d", dc.digits[i]);
        printf("\n");
        printf("%lld", dc.toLongLong());
        printf("\n");
        BigInteger ec = bc % 2;
        printf("\n----------bc % 2;----------\n");
        printf("+++sign%d\n", ec.sign);
        printf("+++size%d\n", ec.size);
        for (int i = 0; i < ec.size; ++i) printf("%d", ec.digits[i]);
        printf("\n");
        printf("%lld", ec.toLongLong());
        printf("\n");
        printf("================================================================\n\n\n\n\n\n\n\n");

    }
}



int main() {

    const int degree = 4;  // 固定長度2^logN bits來表示每一個多項式環的長度
    //const int _2rings = 2;  // one ringpair has 2 rings.
//?CUDA error in cudaMemcpy(d_ringpair, this, sizeof(RingPair), cudaMemcpyHostToDevice) at line 1224: invalid argument
//testBigIntegerOperations<<<1, 1>>>();
//cudaDeviceSynchronize();

    srand(time(0));  // 設定隨機數種子

    // 創建指針來存儲 BigInteger 係數
    BigInteger* h_b_first =
        new BigInteger[degree]();  // 第一個 Ring 的 BigInteger 係數
    BigInteger* h_b_second =
        new BigInteger[degree]();  // 第二個 Ring 的 BigInteger 係數

    for (int j = 0; j < degree; ++j) {
        int sign_first =
            1 - 2 * (rand() % 2);  // 隨機分配符號 (+1 或 -1) 給第一個 Ring
        int size_first = rand() % 6 + 1;  // 隨機大小的 digits (1 ~ 6)
        int* digits_first = new int[size_first]();  // 創建 BigInteger 係數數字
        for (int k = 0; k < size_first; ++k) {
            digits_first[k] = rand() % base;  // 隨機賦值
        }
        h_b_first[j] = BigInteger(sign_first, digits_first,
                                  size_first);  // 用隨機值創建 BigInteger 物件
        // 不需要釋放 digits_first，因為所有權已經交給 h_b_first[j]

        int sign_second =
            1 - 2 * (rand() % 2);  // 隨機分配符號 (+1 或 -1) 給第二個 Ring
        int size_second = rand() % 6 + 1;  // 隨機大小的 digits (1 ~ 6)
        int* digits_second = new int[size_second]();  // 創建 BigInteger 係數數字
        for (int k = 0; k < size_second; ++k) {
            digits_second[k] = rand() % base;  // 隨機賦值
        }
        h_b_second[j] =BigInteger(sign_second, digits_second,
                                  size_second);  // 用隨機值創建 BigInteger 物件
        // 不需要釋放 digits_second，因為所有權已經交給 h_b_second[j]
    }

  

testBigIntegerOperations<<<1, 1>>>();
cudaError_t err = cudaGetLastError();
if (err != cudaSuccess) {
    printf("CUDA Kernel Launch Error: %s\n", cudaGetErrorString(err));
}
cudaDeviceSynchronize();
 printf("\n\n========================================================================\n\n");

 printf("\n\n========================================================================\n\n");

 printf("\n\n========================================================================\n\n");

    // 清理動態分配的內存
   // delete[] h_b_first;
    //delete[] h_b_second;



    return 0;
}
