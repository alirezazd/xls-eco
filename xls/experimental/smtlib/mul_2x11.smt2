; The following SMT-LIB verifies that a 11-bit multiplier is equivalent
; to CVC4's built in bit-vector multiplication.

(set-logic ALL)

; Declare bit-vectors and proxies for indices
(declare-fun x () (_ BitVec 11))
(declare-fun y () (_ BitVec 11))
(define-fun x0 () (_ BitVec 1) ((_ extract 0 0) x))
(define-fun y0 () (_ BitVec 1) ((_ extract 0 0) y))
(define-fun x1 () (_ BitVec 1) ((_ extract 1 1) x))
(define-fun y1 () (_ BitVec 1) ((_ extract 1 1) y))
(define-fun x2 () (_ BitVec 1) ((_ extract 2 2) x))
(define-fun y2 () (_ BitVec 1) ((_ extract 2 2) y))
(define-fun x3 () (_ BitVec 1) ((_ extract 3 3) x))
(define-fun y3 () (_ BitVec 1) ((_ extract 3 3) y))
(define-fun x4 () (_ BitVec 1) ((_ extract 4 4) x))
(define-fun y4 () (_ BitVec 1) ((_ extract 4 4) y))
(define-fun x5 () (_ BitVec 1) ((_ extract 5 5) x))
(define-fun y5 () (_ BitVec 1) ((_ extract 5 5) y))
(define-fun x6 () (_ BitVec 1) ((_ extract 6 6) x))
(define-fun y6 () (_ BitVec 1) ((_ extract 6 6) y))
(define-fun x7 () (_ BitVec 1) ((_ extract 7 7) x))
(define-fun y7 () (_ BitVec 1) ((_ extract 7 7) y))
(define-fun x8 () (_ BitVec 1) ((_ extract 8 8) x))
(define-fun y8 () (_ BitVec 1) ((_ extract 8 8) y))
(define-fun x9 () (_ BitVec 1) ((_ extract 9 9) x))
(define-fun y9 () (_ BitVec 1) ((_ extract 9 9) y))
(define-fun x10 () (_ BitVec 1) ((_ extract 10 10) x))
(define-fun y10 () (_ BitVec 1) ((_ extract 10 10) y))

; Multiply x by y0, shifting x bits accordingly
(define-fun m0_0 () (_ BitVec 1) (bvand x0 y0))
(define-fun m0_1 () (_ BitVec 1) (bvand x1 y0))
(define-fun m0_2 () (_ BitVec 1) (bvand x2 y0))
(define-fun m0_3 () (_ BitVec 1) (bvand x3 y0))
(define-fun m0_4 () (_ BitVec 1) (bvand x4 y0))
(define-fun m0_5 () (_ BitVec 1) (bvand x5 y0))
(define-fun m0_6 () (_ BitVec 1) (bvand x6 y0))
(define-fun m0_7 () (_ BitVec 1) (bvand x7 y0))
(define-fun m0_8 () (_ BitVec 1) (bvand x8 y0))
(define-fun m0_9 () (_ BitVec 1) (bvand x9 y0))
(define-fun m0_10 () (_ BitVec 1) (bvand x10 y0))
(define-fun m0 () (_ BitVec 11) (concat m0_10 m0_9 m0_8 m0_7 m0_6 m0_5 m0_4 m0_3 m0_2 m0_1 m0_0))

; Multiply x by y1, shifting x bits accordingly
(define-fun m1_1 () (_ BitVec 1) (bvand x0 y1))
(define-fun m1_2 () (_ BitVec 1) (bvand x1 y1))
(define-fun m1_3 () (_ BitVec 1) (bvand x2 y1))
(define-fun m1_4 () (_ BitVec 1) (bvand x3 y1))
(define-fun m1_5 () (_ BitVec 1) (bvand x4 y1))
(define-fun m1_6 () (_ BitVec 1) (bvand x5 y1))
(define-fun m1_7 () (_ BitVec 1) (bvand x6 y1))
(define-fun m1_8 () (_ BitVec 1) (bvand x7 y1))
(define-fun m1_9 () (_ BitVec 1) (bvand x8 y1))
(define-fun m1_10 () (_ BitVec 1) (bvand x9 y1))
(define-fun m1 () (_ BitVec 11) (concat m1_10 m1_9 m1_8 m1_7 m1_6 m1_5 m1_4 m1_3 m1_2 m1_1 #b0))

; Multiply x by y2, shifting x bits accordingly
(define-fun m2_2 () (_ BitVec 1) (bvand x0 y2))
(define-fun m2_3 () (_ BitVec 1) (bvand x1 y2))
(define-fun m2_4 () (_ BitVec 1) (bvand x2 y2))
(define-fun m2_5 () (_ BitVec 1) (bvand x3 y2))
(define-fun m2_6 () (_ BitVec 1) (bvand x4 y2))
(define-fun m2_7 () (_ BitVec 1) (bvand x5 y2))
(define-fun m2_8 () (_ BitVec 1) (bvand x6 y2))
(define-fun m2_9 () (_ BitVec 1) (bvand x7 y2))
(define-fun m2_10 () (_ BitVec 1) (bvand x8 y2))
(define-fun m2 () (_ BitVec 11) (concat m2_10 m2_9 m2_8 m2_7 m2_6 m2_5 m2_4 m2_3 m2_2 #b00))

; Multiply x by y3, shifting x bits accordingly
(define-fun m3_3 () (_ BitVec 1) (bvand x0 y3))
(define-fun m3_4 () (_ BitVec 1) (bvand x1 y3))
(define-fun m3_5 () (_ BitVec 1) (bvand x2 y3))
(define-fun m3_6 () (_ BitVec 1) (bvand x3 y3))
(define-fun m3_7 () (_ BitVec 1) (bvand x4 y3))
(define-fun m3_8 () (_ BitVec 1) (bvand x5 y3))
(define-fun m3_9 () (_ BitVec 1) (bvand x6 y3))
(define-fun m3_10 () (_ BitVec 1) (bvand x7 y3))
(define-fun m3 () (_ BitVec 11) (concat m3_10 m3_9 m3_8 m3_7 m3_6 m3_5 m3_4 m3_3 #b000))

; Multiply x by y4, shifting x bits accordingly
(define-fun m4_4 () (_ BitVec 1) (bvand x0 y4))
(define-fun m4_5 () (_ BitVec 1) (bvand x1 y4))
(define-fun m4_6 () (_ BitVec 1) (bvand x2 y4))
(define-fun m4_7 () (_ BitVec 1) (bvand x3 y4))
(define-fun m4_8 () (_ BitVec 1) (bvand x4 y4))
(define-fun m4_9 () (_ BitVec 1) (bvand x5 y4))
(define-fun m4_10 () (_ BitVec 1) (bvand x6 y4))
(define-fun m4 () (_ BitVec 11) (concat m4_10 m4_9 m4_8 m4_7 m4_6 m4_5 m4_4 #b0000))

; Multiply x by y5, shifting x bits accordingly
(define-fun m5_5 () (_ BitVec 1) (bvand x0 y5))
(define-fun m5_6 () (_ BitVec 1) (bvand x1 y5))
(define-fun m5_7 () (_ BitVec 1) (bvand x2 y5))
(define-fun m5_8 () (_ BitVec 1) (bvand x3 y5))
(define-fun m5_9 () (_ BitVec 1) (bvand x4 y5))
(define-fun m5_10 () (_ BitVec 1) (bvand x5 y5))
(define-fun m5 () (_ BitVec 11) (concat m5_10 m5_9 m5_8 m5_7 m5_6 m5_5 #b00000))

; Multiply x by y6, shifting x bits accordingly
(define-fun m6_6 () (_ BitVec 1) (bvand x0 y6))
(define-fun m6_7 () (_ BitVec 1) (bvand x1 y6))
(define-fun m6_8 () (_ BitVec 1) (bvand x2 y6))
(define-fun m6_9 () (_ BitVec 1) (bvand x3 y6))
(define-fun m6_10 () (_ BitVec 1) (bvand x4 y6))
(define-fun m6 () (_ BitVec 11) (concat m6_10 m6_9 m6_8 m6_7 m6_6 #b000000))

; Multiply x by y7, shifting x bits accordingly
(define-fun m7_7 () (_ BitVec 1) (bvand x0 y7))
(define-fun m7_8 () (_ BitVec 1) (bvand x1 y7))
(define-fun m7_9 () (_ BitVec 1) (bvand x2 y7))
(define-fun m7_10 () (_ BitVec 1) (bvand x3 y7))
(define-fun m7 () (_ BitVec 11) (concat m7_10 m7_9 m7_8 m7_7 #b0000000))

; Multiply x by y8, shifting x bits accordingly
(define-fun m8_8 () (_ BitVec 1) (bvand x0 y8))
(define-fun m8_9 () (_ BitVec 1) (bvand x1 y8))
(define-fun m8_10 () (_ BitVec 1) (bvand x2 y8))
(define-fun m8 () (_ BitVec 11) (concat m8_10 m8_9 m8_8 #b00000000))

; Multiply x by y9, shifting x bits accordingly
(define-fun m9_9 () (_ BitVec 1) (bvand x0 y9))
(define-fun m9_10 () (_ BitVec 1) (bvand x1 y9))
(define-fun m9 () (_ BitVec 11) (concat m9_10 m9_9 #b000000000))

; Multiply x by y10, shifting x bits accordingly
(define-fun m10_10 () (_ BitVec 1) (bvand x0 y10))
(define-fun m10 () (_ BitVec 11) (concat m10_10 #b0000000000))

; Add all m bit-vectors to create mul
(define-fun mul () (_ BitVec 11) (bvadd m10 m9 m8 m7 m6 m5 m4 m3 m2 m1 m0))

; Assert and solve
(assert (not (= mul (bvmul x y))))
(check-sat)
