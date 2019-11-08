## 1、depthwise_conv的实现
<pre name="code" class="c">
int8_t arm_depthwise_conv(const int8_t *input,                       //输入向量指针
                          const uint16_t input_h,                    //输入向量高
                          const uint16_t input_w,                    //输入向量宽
                          const uint16_t input_c,                    //输入向量channel数量
                          const int8_t *kernel,                      //卷积核指针
                          const uint16_t c_mult,                     //multiplier
                          const uint16_t stride,                     //stride只为1或2
                          int8_t *output,                            //输出
                          uint8_t inp_fbit,                          //输入小数位数
                          uint8_t ker_fbit,                          //kernel卷积位数
                          uint8_t out_fbit);                         //输出小数位数
</pre>
depthwise convolution函数\
input所指向空间的大小为input_h*input_w*input_c\
kernel所指向空间的大小为3*3*input_c*c_multi\
output的尺寸为input的尺寸除以stride，stride只会为1或者2\
output的大小为input_h/stride * input_w/stride * input_c * c_mult\
input的维度顺序为HWC，output也是HWC\
fbit为定点小数位数\
如果执行成功反回0，否则返回1

卷积核与输入相乘得到int32_t,然后用quantize函数量化为int8_t



## 2、pointwise_conv的实现
<pre name="code" class="c">
int8_t arm_pointwise_conv(const int8_t *input,                      //输入向量指针
                          const uint16_t input_h,                   //输入向量高
                          const uint16_t input_w,                   //输入向量宽
                          const uint16_t input_c,                   //输入向量channel
                          const int8_t *kernel,                     //卷积核指针
                          const uint16_t output_c,                  //输出channel大小
                          int8_t *output,                           //输出指针
                          uint8_t inp_fbit,                         //输入小数位数
                          uint8_t ker_fbit,                         //kernel卷积位数
                          uint8_t out_fbit);                        //输出小数位数
</pre>
pointwise卷积就是一个普通的1x1卷积\
input所指向空间的大小为input_h*input_w*input_c\
kernel所指向空间的大小为input_c*output_c\
stride默认等于1\
output的大小为input_h * input_w * output_c\
input的维度顺序为HWC，output也是HWC\
fbit为定点小数位数\
如果执行成功反回0，否则返回1

卷积核与输入相乘得到int32_t,然后用quantize函数量化为int8_t


## 3、batchnorm的实现
<pre name="code" class="c">
int8_t batchnorm(const int8_t *input,                           //输入向量指针 
                 const uint16_t input_h,                        //输入向量高
                 const uint16_t input_w,                        //输入向量宽
                 const uint16_t input_c,                        //输入向量channel
                 const int8_t *mean,                            
                 const int8_t *var,                             
                 const int8_t *shift,
                 const int8_t *scale,
                 int8_t *output,                                //输出指针
                 uint8_t inp_fbit,                              //输入小数位数
                 uint8_t bn_fbit,                               //mean var shift scale小数位数
                 uint8_t out_fbit);                             //输出小数位数
</pre>
batchnorm层\
input大小为input_h*input_w*input_c\
mean var shift scale所指向空间的的大小都是input_c\
output大小和input相同\
fbit为定点小数位数\
如果执行成功反回0，否则返回1

参数与输入相乘得到int32_t,然后用quantize函数量化为int8_t

## 4、quantize实现
<pre name="code" class="c">
int8_t quantize(const int32_t a,                               //输入
                uint8_t inp_fbit,                              //输入小数位数
                uint8_t out_fbit);                             //输出小数位数
</pre>
输入32位符号数，需要量化成8位符号数;