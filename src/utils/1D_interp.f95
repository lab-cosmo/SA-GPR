subroutine splint(n,x,y,m,xx,yy)

! Given the arrays x and y(x) of dimension n, 
! interpolate with a cubic-spline at the xx value and compute yy(xx)

implicit none
real*8, dimension(n) :: x,y,b,c,d
real*8:: xx, yy,dx 
real*8:: error, errav
integer i,j,k,n,m
double precision f, ispline

! 1: compute spline coefficients

call spline (x,y,b,c,d,n) 

! 2: interpolation at xx points

if(xx <= x(1)) then
  yy = y(1)
  return
end if
if(xx >= x(n)) then
  yy = y(n)
  return
end if

!  binary search for i, such that x(i) <= u <= x(i+1)

i = 1
j = n+1
do while (j > i+1)
  k = (i+j)/2
  if(xx < x(k)) then
    j=k
    else
    i=k
   end if
end do

!  evaluate cucic-spline interpolation

dx = xx - x(i)
yy = y(i) + dx*(b(i) + dx*(c(i) + dx*d(i)))

return
end

!----------------------------------------------------------------------------------
subroutine spline (x,y,b,c,d,n)

! Calculate the coefficients b(i), c(i), and d(i) for a cubic spline interpolation
! x = the arrays of data abscissas (in strictly increasing order)
! y = the arrays of data ordinates
! n = size of the arrays x and y

implicit none
integer n
double precision x(n), y(n), b(n), c(n), d(n)
integer i, j, gap
double precision h

gap = n-1
! check input
if ( n < 2 ) return
if ( n < 3 ) then
  b(1) = (y(2)-y(1))/(x(2)-x(1))   ! linear interpolation
  c(1) = 0.
  d(1) = 0.
  b(2) = b(1)
  c(2) = 0.
  d(2) = 0.
  return
end if

! 1: preparation

d(1) = x(2) - x(1)
c(2) = (y(2) - y(1))/d(1)
do i = 2, gap
  d(i) = x(i+1) - x(i)
  b(i) = 2.0*(d(i-1) + d(i))
  c(i+1) = (y(i+1) - y(i))/d(i)
  c(i) = c(i+1) - c(i)
end do

! 2: end conditions 

b(1) = -d(1)
b(n) = -d(n-1)
c(1) = 0.0
c(n) = 0.0
if(n /= 3) then
  c(1) = c(3)/(x(4)-x(2)) - c(2)/(x(3)-x(1))
  c(n) = c(n-1)/(x(n)-x(n-2)) - c(n-2)/(x(n-1)-x(n-3))
  c(1) = c(1)*d(1)**2/(x(4)-x(1))
  c(n) = -c(n)*d(n-1)**2/(x(n)-x(n-3))
end if

! 3: forward elimination 

do i = 2, n
  h = d(i-1)/b(i-1)
  b(i) = b(i) - h*d(i-1)
  c(i) = c(i) - h*c(i-1)
end do

! 4: back substitution

c(n) = c(n)/b(n)
do j = 1, gap
  i = n-j
  c(i) = (c(i) - d(i)*c(i+1))/b(i)
end do

! 5: compute spline coefficients

b(n) = (y(n) - y(gap))/d(gap) + d(gap)*(c(gap) + 2.0*c(n))
do i = 1, gap
  b(i) = (y(i+1) - y(i))/d(i) - d(i)*(c(i+1) + 2.0*c(i))
  d(i) = (c(i+1) - c(i))/d(i)
  c(i) = 3.*c(i)
end do
c(n) = 3.0*c(n)
d(n) = d(n-1)

return
end

