lin
```
clc;
clear;
x = [1 1 1 1];
h = [1 2 3 4];
m = length(x);
n = length(h); 
a=0:1:m-1;
b=0:1:n-1;
subplot(3,1,1);
plot2d3(a,x);
xlabel('Time'); 
ylabel('Amplitude');
title('Graphical Representation of Input Signal X');
subplot(3,1,2);
plot2d3(b,h);
xlabel('Time'); 
ylabel('Amplitude');
title('Graphical Representation of Impulse Signal h');
for i = 1: n+m-1
conv_sum = 0;
for j = 1:i
if (((i-j+1) <= n)&(j <=m))
conv_sum = conv_sum + x(j)*h(i-j+1);

end;
y(i) = conv_sum;
end;
end;
disp(y,'Convolution Sum using Direct Formula Method = ')
subplot(3,1,3);
plot2d3(y)
title('Graphical Representation of output Signal y');
```
cir
```clc; 
clear;
x=[1 1 1 1];
n1=0:1:length(x)-1;
subplot(3,1,1);
plot2d3(n1,x);
xlabel('time');
ylabel('amplitude');
title('input sequence');
h=[1 2 3];
n2=0:1:length(h)-1;
subplot(3,1,2);
plot2d3(n2,h);
xlabel('time');
ylabel('amplitude');
title('impulse sequence');
N1=length(x);
N2=length(h);
N=max(N1,N2);
N3=N1-N2;
if(N3>0)
h=[h,zeros(1,N3)];
else
x=[x,zeros(1,abs(N3))];
end
disp(x)
disp(h)
for n=1:N
y(n)=0;
for i=1:N
j=n-i+1;
if(j<=0)
j=N+j;
end
y(n)=y(n)+x(i)*h(j);
end
end
disp(y)
n=0:N-1;
subplot(3,1,3);
plot2d3(n,y);
xlabel('time');
ylabel('amplitude');
title('circular convolution');
```
dft
```clc;
clear;
xn=[1 2 3 4 4 3 2 1];
n1=0:1:length(xn)-1;
subplot(3,1,1);
plot2d3(n1,xn);
xlabel('Time n');
ylabel('Amplitude xn');
title('Input Sequence');
j=sqrt(-1);
N=length(xn);
Xk=zeros(1,N);
for k=0:N-1
for n=0:N-1
Xk(k+1)=Xk(k+1)+xn(n+1)*exp((-j*2*%pi*k*n)/N);
end

end
disp(Xk)
K1=0:1:length(Xk)-1;
magnitude=abs(Xk)
subplot(3,1,2);
plot2d3(K1,magnitude);
xlabel('frequency(Hz)');
ylabel('magnitude(gain)');
title('magnitude spectrum');
angle = atan(imag(Xk),real(Xk))
subplot(3,1,3);
plot2d3(K1,angle);
xlabel('frequency(Hz)');
ylabel('Phase');
title('Phase spectrum');
```
fft
```
clear;
clc;
close;
xn = [1 2 3 4 4 3 2 1]
n1=0:1:length(xn)-1;
subplot(2,2,1);
plot2d3(n1,xn);
xlabel('Time n');
ylabel('Amplitude');
title('Input Sequence');
Xk = fft(xn);
K1=0:1:length(Xk)-1;
magnitude=abs(Xk)
subplot(2,2,2);
plot2d3(K1,magnitude);
xlabel('frequency(Hz)');
ylabel('magnitude(gain)');
title('magnitude spectrum');
angle = atan(imag(Xk),real(Xk))
subplot(2,2,3);
plot2d3(K1,angle);
xlabel('frequency(Hz)');
ylabel('Phase');
title('Phase spectrum')
y= ifft(Xk)
n2=0:1:length(y)-1;
subplot(2,2,4)
plot2d3(n2,y)
xlabel('Time n');
ylabel('Amplitude');
title('Inverse FFT OF X(K)');
```
but
```clc ;
close ;
wp=input('Enter the pass band frequency (Radians )= ' );
ws=input('Enter the stop band frequency (Radians )= ' );
alphap=input( ' Enter the pass band attenuation (dB)=' );
alphas=input( ' Enter the stop band attenuation(dB)=' );
T=input('Enter the Value of sampling Time=');
//Pre warping- Bilinear Transformation
omegap=(2/T)*tan(wp/2);
disp(omegap,'omegap=');
omegas=(2/T)*tan(ws/2);
disp(omegas,'omegas=');
//Order of the filter
N=log10(((10^(0.1*alphas))-1)/((10^(0.1*alphap))-1))/(2*log10(omegas/omegap));
disp(N,'N=');
N=ceil(N);
disp(N,'Round off value of N=');
//Cut off frequency
omegac=omegap/(((10^(0.1*alphap)) -1)^(1/(2* N)));
disp(omegac,'omegac=');
disp('Normalised Analog LPF Transfer function H(S)=');
hs_Normalised = analpf(N,'butt',[0,0],1);
disp(hs_Normalised);
disp('Analog LPF Transfer function H(S)=');
hs= analpf(N,'butt',[0,0],omegac);
disp(hs);
z=poly(0,'z');//Defining variable z
Hz=horner(hs,(2/ T)*((z -1)/(z+1)))// Bilinear Transformation
disp('Digital LPF Transfer function H(Z)=');
disp(Hz);
HW=frmag(Hz,512); // Frequency response
w=0:%pi/511:%pi ;
plot(w/%pi,abs(HW));
xlabel(' Normalized Digital Frequency w');
ylabel('Magnitude ');
title(' Frequency Response of Butterworth IIR LPF');
```
che
```
clc ;
close ;
wp=input('Enter the pass band frequency (Radians )= ' );
ws=input('Enter the stop band frequency (Radians )= ' );
alphap=input( ' Enter the pass band attenuation (dB)=' );
alphas=input( ' Enter the stop band attenuation(dB)=' );
T=input('Enter the Value of sampling Time=');
//Pre warping- Bilinear Transformation
omegap=(2/T)*tan(wp/2);
disp(omegap,'omegap=');
omegas=(2/T)*tan(ws/2);
disp(omegas,'omegas=');
//Order of the filter
N=acosh(sqrt(((10^(0.1*alphas))-1)/((10^(0.1*alphap))-1)))/(acosh(omegas/omegap));
disp(N,'N=');
N=ceil(N);
disp(N,'Round off value of N=');
//Cut off frequency
omegac=omegap/(((10^(0.1*alphap)) -1)^(1/(2* N)));
disp(omegac,'omegac=');
Epsilon = sqrt ((10^(0.1*alphap))-1);
disp(Epsilon,'Epsilon=');
[pols ,gn] = zpch1(N, Epsilon,omegap );
disp(gn,'Gain');
disp(pols,'Poles');
hs=poly(gn,'s','coeff')/real(poly(pols,'s'));
disp(hs,'Analog Low pass Chebyshev Filter Transfer function');
z=poly(0,'z');//Defining variable z
Hz=horner(hs,(2/ T)*((z -1)/(z+1)))// Bilinear Transformation
disp(Hz,'Digital LPF Transfer function H(Z)=');
HW=frmag(Hz,512); // Frequency response
w=0:%pi/511:%pi ;
plot(w/%pi,abs(HW));
xlabel(' Normalized Digital Frequency w');
ylabel('Magnitude ');
title(' Frequency Response of Chebyshev IIR LPF');
```
lpf
```
clc ;
close ;
M=input('Enter the Odd Filter Length =');
Wc=input('Enter the Digital Cut off frequency =');
alpha= (M -1)/2 // Center Value
for n = 1:M
if (n ==alpha+1)
hd(n) = Wc/ %pi ;
else
hd(n) = sin(Wc *((n -1)-alpha)) /(((n -1)-alpha)*%pi);
end
end
// Rectangular Window
for n = 1:M
W(n) = 1;
end
//Windowing filter coefficients
h = hd.*W;
disp(h,'Filter Coefficients are')
[hzm,fr]= frmag (h,256) ;
subplot(2 ,1 ,1)
plot(2*fr, hzm)
xlabel( ' Normalized Digital Frequency w');
ylabel( 'Magnitude ');
title( ' Frequency Response of FIR LPF using Rectangular Window ')
hzm_dB = 20* log10 (hzm);
subplot (2 ,1 ,2);
plot(2*fr , hzm_dB);
xlabel( ' Normalized Digital Frequency W' );
ylabel( 'Magnitude in dB');
title('Frequency Response of FIR LPF using Rectangular Window');
```
hpf
```
clc ;
close ;
M=input('Enter the Odd Filter Length =');
Wc=input('Enter the Digital Cut off frequency =');
alpha= (M -1)/2 // Center Value
for n = 1:M
if (n ==alpha+1)
hd(n)=1-Wc/ %pi ;
else
hd(n) =-sin(Wc *((n -1)-alpha)) /(((n -1)-alpha)*%pi);
end
end
// Rectangular Window
for n = 1:M
W(n) =1;
end
//Windowing filter coefficients
h = hd.*W;
disp(h,'Filter Coefficients are')
[hzm,fr]= frmag (h,256) ;
subplot(2 ,1 ,1)
plot(2*fr, hzm)
xlabel( ' Normalized Digital Frequency w');
ylabel( 'Magnitude ');
title( ' Frequency Response of FIR LPF using Rectangular Window ')
hzm_dB = 20* log10 (hzm);
subplot (2 ,1 ,2);
plot(2*fr , hzm_dB);
xlabel( ' Normalized Digital Frequency W' );
ylabel( 'Magnitude in dB');
title('Frequency Response of FIR HPF using Rectangular Window');
```
bpf
```
clc ;
close ;
M=input('Enter the Odd Filter Length =');
Wc=input('Enter the Digital Cut off frequency =');
Wc2=Wc(2);
Wc1=Wc(1);
alpha= (M -1)/2 // Center Value
for n = 1:M
if (n ==alpha+1)
hd(n) =(Wc2-Wc1)/%pi ;
else
hd(n) =((sin(Wc2 *((n -1)-alpha)))-(sin(Wc1 *((n -1)-alpha))))/(((n -1)-alpha)*%pi);
end
end
// Rectangular Window
for n = 1:M
W(n) =1;
end
//Windowing filter coefficients
h = hd.*W;
disp(h,'Filter Coefficients are')
[hzm,fr]= frmag (h,256) ;
subplot(2 ,1 ,1)
plot(2*fr, hzm)
xlabel( ' Normalized Digital Frequency w');
ylabel( 'Magnitude ');
title( ' Frequency Response of FIR BPF using Rectangular Window ')
hzm_dB = 20* log10 (hzm);
subplot (2 ,1 ,2);
plot(2*fr , hzm_dB);
xlabel( ' Normalized Digital Frequency W' );
ylabel( 'Magnitude in dB');
title('Frequency Response of FIR BPF using Rectangular Window');
```
bsf
```
clc ;
close ;
M=input('Enter the Odd Filter Length =');
Wc=input('Enter the Digital Cut off frequency =');
Wc2=Wc(2);
Wc1=Wc(1);
alpha= (M -1)/2 // Center Value
for n = 1:M
if (n ==alpha+1)
hd(n) =1-((Wc2-Wc1)/%pi) ;
else
hd(n) =((sin(Wc1 *((n -1)-alpha)))-(sin(Wc2 *((n -1)-alpha))))/(((n -1)-alpha)*%pi);
end
end
// Rectangular Window
for n = 1:M
W(n) =1;
end
//Windowing filter coefficients
h = hd.*W;
disp(h,'Filter Coefficients are')
[hzm,fr]= frmag (h,256) ;
subplot(2 ,1 ,1)
plot(2*fr, hzm)
xlabel( ' Normalized Digital Frequency w');
ylabel( 'Magnitude ');
title( ' Frequency Response of FIR BSF using Rectangular Window ')
hzm_dB = 20* log10 (hzm);
subplot (2 ,1 ,2);
plot(2*fr , hzm_dB);
xlabel( ' Normalized Digital Frequency W' );
ylabel( 'Magnitude in dB');
title('Frequency Response of FIR BSF using Rectangular Window');
```
mul
```
clear;
clc;
close;
n = 0:%pi/50:2*%pi;
x = sin(%pi*n); //original signal
M=input('Enter the downsampling factor');
L=input('Enter the upsampling factor');
//Down Sampling
downsampling_x = x(1:M:length(x));
disp(x,'Input signal x(n)=');
disp(downsampling_x,'Downsampled Signal');
figure(1);
subplot(2,1,1)
plot2d3(1:length(x),x);
xtitle('original singal')
subplot(2,1,2)
plot2d3(1:length(downsampling_x),downsampling_x);
xtitle('Downsampled Signal by a factor of M');
//Upsampling
upsampling_x=[];
for i=1:length(x)
upsampling_x(1,L*i)=x(i);
end
disp(x,'Input signal x(n)=');
disp(upsampling_x,'Upsampled Signal');
figure(2);
subplot(2,1,1);
plot2d3(x);
title('original signal');
subplot(2,1,2);
plot2d3(upsampling_x);
title('Upsampled Signal by a factor of L');
```
