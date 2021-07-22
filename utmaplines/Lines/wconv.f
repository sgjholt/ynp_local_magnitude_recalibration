      program wconv
c
c	Converts wusbomap format to qplot format
c
      real lat,latmin,lon,lonmin,xlat,xlon
      character*40 name,infil,outfil
      write(6,*) 'Enter input line file: '
      read(5,12) infil
      write(6,*) 'Enter output line file: '
      read(5,12) outfil
 12   format(a)
      open(1,file=infil)
      open(2,file=outfil)
      rewind(1)
      rewind(2)
  10  format(1x,f7.4,1x,f7.4,f8.4,1x,f7.4)
  20  format(f8.4,1x,f9.4)
  100 read(1,10,end=99) lat,latmin,lon,lonmin
      xlat = lat + latmin/60.
      xlon = lon + lonmin/60.
      write(2,20) xlat,xlon
      goto 100
  99  close(1)
      close(2)
      stop
      end
