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

subroutine get_skernel(lval,lcut,mcut,nspecies,nnmax,natmax,npoints,nneigh,CG2,efact,sph_i6,length,nat,skernel)
implicit none

 integer lval,lcut,mcut,nspecies,nnmax,natmax,npoints,nneigh(npoints,natmax,nspecies),nat(npoints)
 integer i,j,nm,ii,jj,iii,jjj,ix,l,m1,m2
 real*8 CG2(lcut+1,lcut+1,mcut,mcut,2*lval+1,2*lval+1),efact(npoints,natmax,nspecies,nnmax)
 real*8 length(npoints,natmax,nspecies,nnmax)
 real*8 di(0:lcut),sph_in(nnmax,nnmax,0:lcut)
 complex*16 sph_i6(npoints,natmax,nspecies,nnmax,0:lcut,2*lcut+1)
 complex*16 skernel(npoints,npoints,natmax,natmax,2*lval+1,2*lval+1),ISOAP(nspecies,0:lcut,mcut,mcut)

!f2py intent(in) lval,lcut,mcut,nspecies,nnmax,natmax,npoints,nneigh,CG2,efact,sph_i6,length,nat
!f2py intent(out) skernel
!f2py depend(lcut) CG2,sph_i6,ISOAP
!f2py depend(mcut) CG2,ISOAP
!f2py depend(lval) CG2,skernel
!f2py depend(natmax) nneigh,efact,sph_i6,length,skernel
!f2py depend(nspecies) nneigh,efact,sph_i6,length,ISOAP
!f2py depend(nnmax) efact,sph_i6,length
!f2py depend(npoints) nneigh,efact,sph_i6,length,nat,skernel

 skernel(:,:,:,:,:,:) = 0.d0

 do i=0,npoints-1
  do j=0,i
   do ii=0,nat(i+1)-1
    do jj=0,nat(j+1)-1
     ISOAP(:,:,:,:) = 0.d0
     do ix=0,nspecies-1
      ! Calculate spherical Bessel functions.
      do iii=0,nneigh(i+1,ii+1,ix+1)-1
       do jjj=0,nneigh(j+1,jj+1,ix+1)-1
        call sphi(lcut,length(i+1,ii+1,ix+1,iii+1)*length(j+1,jj+1,ix+1,jjj+1),nm,sph_in(iii+1,jjj+1,:),di)
       enddo
      enddo
      ! Fill ISOAP array.
      do l=0,lcut
       do m1=0,2*lcut
        do m2=0,2*lcut
         do iii=0,nneigh(i+1,ii+1,ix+1)-1
          do jjj=0,nneigh(j+1,jj+1,ix+1)-1
           ISOAP(ix+1,l,m1+1,m2+1) = ISOAP(ix+1,l,m1+1,m2+1) + &
     &      efact(i+1,ii+1,ix+1,iii+1)*efact(j+1,jj+1,ix+1,jjj+1)*sph_in(iii+1,jjj+1,l)* &
     &      sph_i6(i+1,ii+1,ix+1,iii+1,l,m1+1)*dconjg(sph_i6(j+1,jj+1,ix+1,jjj+1,l,m2+1))
          enddo
         enddo
        enddo
       enddo
      enddo
     enddo
     call fill_spectra(lval,lcut,mcut,nspecies,ISOAP,CG2,skernel(i+1,j+1,ii+1,jj+1,:,:))
     if (i.ne.j) skernel(j+1,i+1,jj+1,ii+1,:,:) = dconjg(transpose(skernel(i+1,j+1,ii+1,jj+1,:,:)))
    enddo
   enddo
  enddo
 enddo

end subroutine

!***************************************************************
!* REFERENCE: "Fortran Routines for Computation of Special     *
!*             Functions,                                      *
!*             jin.ece.uiuc.edu/routines/routines.html".       *
!*                                                             *
!*                           F90 Release By J-P Moreau, Paris. *
!*                                  (www.jpmoreau.fr)          *
!***************************************************************


        SUBROUTINE SPHI(N,X,NM,SI,DI)

!      ========================================================
!      Purpose: Compute modified spherical Bessel functions
!               of the first kind, in(x) and in'(x)
!      Input :  x --- Argument of in(x)
!               n --- Order of in(x) ( n = 0,1,2,... )
!      Output:  SI(n) --- in(x)
!               DI(n) --- in'(x)
!               NM --- Highest order computed
!      Routines called:
!               MSTA1 and MSTA2 for computing the starting
!               point for backward recurrence
!      ========================================================

        IMPLICIT DOUBLE PRECISION (A-H,O-Z)
        DIMENSION SI(0:N),DI(0:N)
        NM=N
        IF (DABS(X).LT.1.0D-100) THEN
           DO 10 K=0,N
              SI(K)=0.0D0
10            DI(K)=0.0D0
           SI(0)=1.0D0
           DI(1)=0.333333333333333D0
           RETURN
        ENDIF
        SI(0)=DSINH(X)/X
        SI(1)=-(DSINH(X)/X-DCOSH(X))/X
        SI0=SI(0)
        IF (N.GE.2) THEN
           M=MSTA1(X,200)
           IF (M.LT.N) THEN
              NM=M
           ELSE
              M=MSTA2(X,N,15)
           ENDIF
           F0=0.0D0
           F1=1.0D0-100
           DO 15 K=M,0,-1
              F=(2.0D0*K+3.0D0)*F1/X+F0
              IF (K.LE.NM) SI(K)=F
              F0=F1
15            F1=F
           CS=SI0/F
           DO 20 K=0,NM
20            SI(K)=CS*SI(K)
        ENDIF
        DI(0)=SI(1)
        DO 25 K=1,NM
25         DI(K)=SI(K-1)-(K+1.0D0)/X*SI(K)
        RETURN
        END


        INTEGER FUNCTION MSTA1(X,MP)

!      ===================================================
!      Purpose: Determine the starting point for backward  
!               recurrence such that the magnitude of    
!               Jn(x) at that point is about 10^(-MP)
!      Input :  x     --- Argument of Jn(x)
!               MP    --- Value of magnitude
!      Output:  MSTA1 --- Starting point   
!      ===================================================

        IMPLICIT DOUBLE PRECISION (A-H,O-Z)
        A0=DABS(X)
        N0=INT(1.1*A0)+1
        F0=ENVJ(N0,A0)-MP
        N1=N0+5
        F1=ENVJ(N1,A0)-MP
        DO 10 IT=1,20             
           NN=N1-(N1-N0)/(1.0D0-F0/F1)                  
           F=ENVJ(NN,A0)-MP
           IF(ABS(NN-N1).LT.1) GO TO 20
           N0=N1
           F0=F1
           N1=NN
 10        F1=F
 20     MSTA1=NN
        RETURN
        END


        INTEGER FUNCTION MSTA2(X,N,MP)

!      ===================================================
!      Purpose: Determine the starting point for backward
!               recurrence such that all Jn(x) has MP
!               significant digits
!      Input :  x  --- Argument of Jn(x)
!               n  --- Order of Jn(x)
!               MP --- Significant digit
!      Output:  MSTA2 --- Starting point
!      ===================================================

        IMPLICIT DOUBLE PRECISION (A-H,O-Z)
        A0=DABS(X)
        HMP=0.5D0*MP
        EJN=ENVJ(N,A0)
        IF (EJN.LE.HMP) THEN
           OBJ=MP
           N0=INT(1.1*A0)
        ELSE
           OBJ=HMP+EJN
           N0=N
        ENDIF
        F0=ENVJ(N0,A0)-OBJ
        N1=N0+5
        F1=ENVJ(N1,A0)-OBJ
        DO 10 IT=1,20
           NN=N1-(N1-N0)/(1.0D0-F0/F1)
           F=ENVJ(NN,A0)-OBJ
           IF (ABS(NN-N1).LT.1) GO TO 20
           N0=N1
           F0=F1
           N1=NN
10         F1=F
20      MSTA2=NN+10
        RETURN
        END

        REAL*8 FUNCTION ENVJ(N,X)
        DOUBLE PRECISION X
        ENVJ=0.5D0*DLOG10(6.28D0*N)-N*DLOG10(1.36D0*X/N)
        RETURN
        END
