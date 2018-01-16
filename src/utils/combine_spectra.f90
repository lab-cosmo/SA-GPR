! COMBINE POWER SPECTRA L=0

subroutine combine_spectra(lcut,mcut,nspecies,ISOAP,divfac,PS)
implicit none

 integer lcut,mcut,l,im,ik,nspecies,ix,iy
 real*8 divfac(lcut+1)
 complex*16 ISOAP(nspecies,lcut+1,mcut,mcut)
 complex*16 PS, PSS, PKM, dcix
!f2py intent(in) lcut,mcut,nspecies,ISOAP,divfac
!f2py intent(out) PS
!f2py depend(lcut) ISOAP, divfac
!f2py depend(mcut) ISOAP,PS
!f2py depend(lval) PS
!f2py depend(nspecies) ISOAP
 PS = 0.d0
 do l=0,lcut
  PSS = 0.d0
  do im=0,2*l
   do ik=0,2*l
    do ix=0,nspecies-1
     dcix = dconjg(ISOAP(ix+1,l+1,im+1,ik+1))
     PKM = 0.0d0
     do iy=0,nspecies-1 
      PKM = PKM + ISOAP(iy+1,l+1,im+1,ik+1) 
     enddo
     PSS = PSS + PKM * dcix
    enddo
   enddo
  enddo
  PS = PS + PSS * divfac(l+1)
 enddo
end subroutine
