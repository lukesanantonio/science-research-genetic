;; When this code is run repeatedly, it will walk up a vertical column and
;; change state to match the height of a composite function. When the top cell
;; in the column is reached,
(if (< abspos_y
       (* gridsize_y                              ; amplitude is max grid height
          (+ (sin (* abspos_x 5))                 ; use a high frequency
             (+ (sin (* abspos_z 5)) (/ pi 2))))) ; 90 deg phase difference

  ;; If the value is larger than our current value, set the voxel to true and
  ;; move up. When we reach the max height, the absolute position y-value will
  ;; always be larger than the height function, meaning the else-expression will
  ;; be run on the next iteration. However, the code will be penalized for
  ;; extending out of the bounds of the world.
  (do (set true) (move 0 1 0))

  ;; If the value is smaller, set the voxel to false. If we are below the max y
  ;; grid value, keep going up.
  (do (set false) (if (< abspos_y (- gridsize_y 1))
                        (move 0 1 0)
                        (move (uniform) (- gridsize_y) (uniform)))))
