# Ivans_QATM_GPU
GPU version of my QATM project. This is CUDA version

<b>I do not allow to copy any line of this code for anybody that did not ever studied or worked in MTSU</b>

If you want to try it:
<p>  1) download everything.
<p>  2) run <b>make</b> in your lovely UNIX console.(you need nvcc)
<p>  3) enjoy flying numbers.

My current results on Intel i5-4690 CPU @ 3.50GHz(single core) and GeForce GTX 550 Ti :

I know it is bad, but I never started optimisation, and profiler says, that I use around 5% of compute power.

<p><b>GPU</b> time for mem COPY for 5 functions: 0.0000240000000000
<p><b>GPU</b> time to COPY back results: 0.0000550000000000

<p>[<b>1 iterations</b>]
<p><b>CPU</b> time for 5 functions: 0.0000010000000000
<p><b>GPU</b> time for 5 functions: 0.0000260000000000

<p>[<b>10 iterations</b>]
<p><b>CPU</b> time for 5 functions: 0.0000020000000000
<p><b>GPU</b> time for 5 functions: 0.0000720000000000

<p>[<b>100 iterations</b>]
<p><b>CPU</b> time for 5 functions: 0.0000120000000000
<p><b>GPU</b> time for 5 functions: 0.0006820000000000

<p>[<b>1000 iterations</b>]
<p><b>CPU</b> time for 5 functions: 0.0001250000000000
<p><b>GPU</b> time for 5 functions: 0.0173650000000000

<p>[<b>10000 iterations</b>]
<p><b>CPU</b> time for 5 functions: 0.0012480000000000
<p><b>GPU</b> time for 5 functions: 0.2143430000000000