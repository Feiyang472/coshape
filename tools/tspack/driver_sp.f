      SUBROUTINE TSP_SP(N,X,Y,BV1,BVN,M,XE,HE,HPE,HPPE,YP,
     .                  SIGMA,ITER,IER)
C     Shape-preserving reference: Renka's TSPSI with NCD=2 (C^2), clamped
C     end slopes (IENDC=1, YP(1)=BV1, YP(N)=BVN), non-uniform tension so
C     SIGS selects the minimal per-interval tension that preserves local
C     convexity/monotonicity.  Returns the per-interval SIGMA, knot slopes
C     YP, iteration count, and value/1st/2nd derivative on XE.  This mirrors
C     coshape's shape-preserving TensionSpline exactly.
      INTEGER N, M, ITER, IER
      DOUBLE PRECISION X(N), Y(N), BV1, BVN
      DOUBLE PRECISION XE(M), HE(M), HPE(M), HPPE(M)
      DOUBLE PRECISION YP(N), SIGMA(N)
      INTEGER I, IER1, LWK, NCD, IENDC
      LOGICAL PER, UNIFRM
      DOUBLE PRECISION WK(3*N)
      DOUBLE PRECISION HVAL, HPVAL, HPPVAL
      EXTERNAL HVAL, HPVAL, HPPVAL, TSPSI
      NCD = 2
      IENDC = 1
      PER = .FALSE.
      UNIFRM = .FALSE.
      LWK = 3*N
      DO 5 I = 1, N
        YP(I) = 0.0D0
        SIGMA(I) = 0.0D0
    5 CONTINUE
C     Clamped end slopes are passed to TSPSI via YP(1), YP(N).
      YP(1) = BV1
      YP(N) = BVN
      CALL TSPSI(N,X,Y,NCD,IENDC,PER,UNIFRM,LWK,WK,YP,SIGMA,IER)
      ITER = IER
      IF (IER .LT. 0) RETURN
      DO 10 I = 1, M
        HE(I)   = HVAL  (XE(I),N,X,Y,YP,SIGMA,IER1)
        HPE(I)  = HPVAL (XE(I),N,X,Y,YP,SIGMA,IER1)
        HPPE(I) = HPPVAL(XE(I),N,X,Y,YP,SIGMA,IER1)
   10 CONTINUE
      RETURN
      END
