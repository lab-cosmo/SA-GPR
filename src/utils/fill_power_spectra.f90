! FILL POWER SPECTRA

subroutine fill_spectra(lval,lcut,mcut,nspecies,ISOAP,CG2,PS)
implicit none

 integer lval,lcut,mcut,l,im,ik,l1,iim,iik,nspecies,ix,iy
 complex*16 ISOAP(nspecies,lcut+1,mcut,mcut)
 real*8 CG2(lcut+1,lcut+1,mcut,mcut,2*lval+1,2*lval+1)
 complex*16 PS(2*lval+1,2*lval+1), PKM, dcix
!f2py intent(in) lval,lcut,mcut,nspecies,ISOAP,CG2
!f2py intent(out) PS
!f2py depend(lcut) ISOAP,CG2
!f2py depend(mcut) ISOAP,PS,CG2
!f2py depend(lval) PS,CG2
!f2py depend(nspecies) ISOAP
 PS(:,:) = 0.d0

   do l1=0,lcut
    do l=0,lcut
     do im=0,2*l
      do ik=0,2*l
       do ix=0,nspecies-1
       dcix = dconjg(ISOAP(ix+1,l+1,im+1,ik+1))
       do iim=max(0,lval+im-l-l1),min(2*lval,lval+im-l+l1)
        do iik=max(0,lval+ik-l-l1),min(2*lval,lval+ik-l+l1)
          PKM = 0.0d0
          do iy=0,nspecies-1 
           PKM = PKM + ISOAP(iy+1,l1+1,im-l+l1-iim+lval+1,ik-l+l1-iik+lval+1) 
          enddo
          PS(iim+1,iik+1) = PS(iim+1,iik+1) + PKM  * &
                CG2(l+1,l1+1,im+1,ik+1,iim+1,iik+1) * dcix
         enddo
        enddo
       enddo
      enddo
     enddo
    enddo
   enddo
end subroutine
