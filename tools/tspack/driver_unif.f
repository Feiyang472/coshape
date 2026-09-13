      SUBROUTINE TSP_UNIF(N,X,Y,SIG,ISL1,ISLN,BV1,BVN,M,XE,
     .                    HE,HPE,HPPE,YP,IER)
C     Uniform-tension reference: build a C^2 Hermite tension spline with
C     a single tension factor SIG on every interval (YPC2 solves for the
C     C^2 knot slopes), then evaluate value / 1st / 2nd derivative on XE.
C     This mirrors coshape's uniform-tension TensionSpline exactly, so
C     the two can be compared to floating-point tolerance.
C
C     ISL1/ISLN, BV1/BVN are the YPC2 end-condition flags/values:
C       ISL=1 => clamped (first derivative = BV);  ISL=0 => parabolic fit.
      INTEGER N, ISL1, ISLN, M, IER
      DOUBLE PRECISION X(N), Y(N), SIG, BV1, BVN
      DOUBLE PRECISION XE(M), HE(M), HPE(M), HPPE(M), YP(N)
      INTEGER I, IER1
      DOUBLE PRECISION SIGMA(N), WK(N)
      DOUBLE PRECISION HVAL, HPVAL, HPPVAL
      EXTERNAL HVAL, HPVAL, HPPVAL, YPC2
      DO 5 I = 1, N
        SIGMA(I) = SIG
        YP(I) = 0.0D0
    5 CONTINUE
      CALL YPC2(N,X,Y,SIGMA,ISL1,ISLN,BV1,BVN,WK, YP,IER)
      IF (IER .LT. 0) RETURN
      DO 10 I = 1, M
        HE(I)   = HVAL  (XE(I),N,X,Y,YP,SIGMA,IER1)
        HPE(I)  = HPVAL (XE(I),N,X,Y,YP,SIGMA,IER1)
        HPPE(I) = HPPVAL(XE(I),N,X,Y,YP,SIGMA,IER1)
   10 CONTINUE
      RETURN
      END
