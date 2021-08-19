#!/opt/local/bin/python
import numpy as np
from numpy import *
from matplotlib.pyplot import plot, figure, show, draw, pause, ion, ioff
from matplotlib.pyplot import clf, title, semilogy, imshow
from scipy.integrate import ode
import getopt, sys

xsize = 128
tmax = 300.0
alpha = 0.160
eps = 0.5 
dx = 0.5
dt = 0.03
plotevery = 30
clearevery = 30

try:
  opts, args = getopt.getopt(sys.argv[1:], "i:")
except getopt.GetoptError as err:
  print(err)
  sys.exit(2)

writeout = False
readin = False
if (len(args) > 0): 
  writeout = True
  fileout = args[0]
  print('will write output to', fileout)
for o, a in opts:
  if o == '-i':
    readin = True
    datain = loadtxt(a)
    print('will read input from', a)

clearn = clearevery * plotevery
nt = int(tmax / dt)+1
noise = random.randn(nt, xsize)

def f(t,y, eps=0, a=0.20, dx=0.5):
  y2 = roll(y,-2)
  y1 = roll(y,-1)
  ym1 = roll(y,1)
  ym2 = roll(y,2)
  return (-a * y 
    - (y1 - 2*y + ym1)/(dx*dx)
    - (y2 - 4*y1 + 6*y - 4*ym1 + ym2)/(dx*dx*dx*dx)
    + (y1 - ym1) * (y1 - ym1) / (4 * dx * dx)
    )

y0 = random.randn(xsize)
if readin: y0 = datain[-1]
r = ode(f).set_integrator('lsoda').set_f_params(eps, alpha, dx)
r.set_initial_value(y0, 0)

i = 0
ion()
lines = []
while r.successful() and r.t < tmax:
  yout = r.integrate(r.t+dt) + eps * noise[i]
  if (i%clearn == 0): 
    clf()
    title('xsize, alpha, eps, dx, dt = %d %.3f %.3g %.3f %.3f' 
      % (xsize, alpha, eps, dx, dt))
    print('iteration', i, 'time', round(r.t,3), ' --- ', 100*r.t//tmax,'%')
  if (i%plotevery == 1):
    c = (1.0 - (i % clearn)/clearn, 0,0)
    lines.append(yout)
    plot(yout, linestyle='-', color=c)
    draw()
    pause(0.001)
  i = i+1
yout = r.integrate(r.t+dt)
figure()
semilogy(abs(fft.fft(yout-mean(yout))[0:xsize//2])**2)
title('xsize, alpha, eps, dx, dt = %d %.3f %.3g %.3f %.3f' 
  % (xsize, alpha, eps, dx, dt))
draw()
figure()
img = array(lines)
if readin: img = concatenate([datain, img])
if writeout: savetxt(fileout, img)
vmax = np.mean(img) + 0.95 * (np.max(img) - np.mean(img))
vmin = np.mean(img) - 0.95 * (np.mean(img) - np.min(img))
imshow(img, vmax=vmax, vmin=vmin, origin='lower', cmap='hot', aspect='auto')
title('xsize, alpha, eps, dx, dt = %d %.3f %.3g %.3f %.3f' 
  % (xsize, alpha, eps, dx, dt))
draw()
pause(999)
