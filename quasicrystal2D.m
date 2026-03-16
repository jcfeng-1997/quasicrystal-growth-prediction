clc; clear; close all;

Lx=256; Ly=256; Mx=256; My=256;
dx=Lx/Mx; dy=Ly/My;
x=0+(0:Mx-1)*dx;
y=0+(0:My-1)*dy;
[xx,yy]=meshgrid(x,y);
xix=2*pi*[0:Mx/2-1 -Mx/2:-1]/Lx;
xiy=2*pi*[0:My/2-1 -My/2:-1]/Ly;
[kx,ky]=ndgrid(xix,xiy);
dt=0.1; h=dx; g=0; r=0.0; s=2;
T=500; order=2;
qq=2*cos(pi/12); rr=0.01;
kp=-(s-(2*qq^4+2*qq^2+2*rr^2)*(kx.^2+ky.^2)+...
    (qq^4+4*qq^2+1+rr^2)*(kx.^2+ky.^2).^2-...
    (2*qq^2+2)*(kx.^2+ky.^2).^3+(kx.^2+ky.^2).^4);


switch order
    case 2
        gamma=(2-sqrt(2))/2; lap=1-1/(2*gamma);
        MI=[gamma 0;1-gamma gamma];
        ME=[gamma 0;lap 1-lap];
end
ns=length(MI);


alphas = 1.0;
epsilons = 0:0.01:0.07;


for alpha = alphas
    for epsilon = epsilons

        
        folder_name = sprintf('outdata/alpha_%.2f_epsilon_%.2f', alpha, epsilon);
        if ~exist(folder_name, 'dir')
            mkdir(folder_name);
        end

        
        R = 1*(2*rand(Mx,My)-1);
        ophi = 0+(R-mean(R(:)));

        
        energynum = [];

        
        for n = 1:round(T/dt)
            ophi_hat=fft2(ophi);
            phis=zeros(Mx,My,ns); q=zeros(Mx,My,ns);

            for i=1:ns
                Phip=ophi.^3 - epsilon*ophi - alpha*ophi.^2 + (qq^4+rr^2)*ophi;
                Phir=(1/4*ophi.^4 - epsilon/2*ophi.^2 - alpha/3*ophi.^3 + (qq^4+rr^2)/2*ophi.^2).^r;
                beta=sum(Phip(:)) / sum(Phir(:) + 1e-12); 

                q(:,:,i) = -(Phip - s*ophi) + Phir*beta;
                IM = 0; EX = ME(i,i)*q(:,:,i);

                for j=1:i-1
                    IM = IM + MI(i,j)*phis(:,:,j);
                    EX = EX + ME(i,j)*q(:,:,j);
                end

                phis(:,:,i) = (ophi_hat + dt*(kp.*IM + fft2(EX))) ./ (1 - MI(i,i)*dt*kp);
                ophi = real(ifft2(phis(:,:,i)));
            end

            
            if mod(n,10) == 0
                figure(1); clf;
                pcolor(xx,yy,ophi); shading interp; colormap jet; axis image off;
                fname = sprintf('%s/phi%d.m', folder_name, n/10);
                fid = fopen(fname, 'wt');
                fprintf(fid, '%10.8f\n', ophi);
                fclose(fid);
            end

            end
        end
    end

