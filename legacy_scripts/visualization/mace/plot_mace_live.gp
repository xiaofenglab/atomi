# =========================================================
# MACE live monitor for gnuplot
#
# Usage from wrapper:
#   gnuplot -e "file='results/train.txt'; win=100; refresh=5" plot_mace_live.gp
#
# Inputs:
#   file     = MACE log file
#   win      = number of latest epochs to display (default 100)
#   refresh  = refresh interval in seconds (default 5)
#
# Notes:
# - uses reread for live updating
# - default window is latest 100 epochs
# - panel colors:
#     loss    -> cyan
#     RMSE_F  -> yellow
#     RMSE_E  -> green
#     E vs F  -> magenta
# =========================================================

if (!exists("file")) {
    print "Usage: plotmace <mace_train.log> [window_epochs] [refresh_seconds]"
    exit
}

if (!exists("win")) win = 100
if (win < 1) win = 100

if (!exists("refresh")) refresh = 5
if (refresh < 1) refresh = 5

set term dumb ansi 180 60

# -------------------------
# Helper output files
# -------------------------
datafile = "/tmp/mace_live_epochs.dat"
metafile = "/tmp/mace_live_meta.txt"

# -------------------------
# Parse epoch / initial lines
#
# Output data columns:
#   1 plot_epoch   (Initial = -1)
#   2 true_epoch
#   3 loss
#   4 RMSE_E_meV_per_atom
#   5 RMSE_F_meV_A
# -------------------------
parse_cmd = \
"awk '\
BEGIN{best_loss=1e99; best_e=1e99; best_f=1e99;} \
/INFO: Initial: head:/ { \
  loss=\"\"; e=\"\"; f=\"\"; \
  if (match($0,/loss= *([0-9.eE+-]+)/,a)) loss=a[1]; \
  if (match($0,/RMSE_E_per_atom= *([0-9.eE+-]+)/,b)) e=b[1]; \
  if (match($0,/RMSE_F= *([0-9.eE+-]+)/,c)) f=c[1]; \
  print -1, -1, loss, e, f; \
  if (loss != \"\" && loss+0 < best_loss) best_loss=loss+0; \
  if (e    != \"\" && e+0    < best_e)    best_e=e+0; \
  if (f    != \"\" && f+0    < best_f)    best_f=f+0; \
  init_loss=loss; init_e=e; init_f=f; \
} \
/INFO: Epoch [0-9]+:/ { \
  ep=\"\"; loss=\"\"; e=\"\"; f=\"\"; \
  if (match($0,/Epoch *([0-9]+):/,a)) ep=a[1]; \
  if (match($0,/loss= *([0-9.eE+-]+)/,b)) loss=b[1]; \
  if (match($0,/RMSE_E_per_atom= *([0-9.eE+-]+)/,c)) e=c[1]; \
  if (match($0,/RMSE_F= *([0-9.eE+-]+)/,d)) f=d[1]; \
  print ep, ep, loss, e, f; \
  last_ep=ep; last_loss=loss; last_e=e; last_f=f; \
  if (loss != \"\" && loss+0 < best_loss) best_loss=loss+0; \
  if (e    != \"\" && e+0    < best_e)    best_e=e+0; \
  if (f    != \"\" && f+0    < best_f)    best_f=f+0; \
} \
END{ \
  print \"last_epoch=\" last_ep        > \"" . metafile . "\"; \
  print \"last_loss=\" last_loss      >> \"" . metafile . "\"; \
  print \"last_e=\" last_e            >> \"" . metafile . "\"; \
  print \"last_f=\" last_f            >> \"" . metafile . "\"; \
  print \"init_loss=\" init_loss      >> \"" . metafile . "\"; \
  print \"init_e=\" init_e            >> \"" . metafile . "\"; \
  print \"init_f=\" init_f            >> \"" . metafile . "\"; \
  print \"best_loss=\" best_loss      >> \"" . metafile . "\"; \
  print \"best_e=\" best_e            >> \"" . metafile . "\"; \
  print \"best_f=\" best_f            >> \"" . metafile . "\"; \
}' '" . file . "' > '" . datafile . "'"

system(parse_cmd)

# -------------------------
# Pull metadata from log
# -------------------------
fname       = system("basename " . file)
foundation  = system("awk -F'INFO: ' '/Using foundation model/{print $2; exit}' " . file)
train_valid = system("awk -F'INFO: ' '/Total number of configurations:/{print $2; exit}' " . file)
neighbors   = system("awk -F'INFO: ' '/Average number of neighbors:/{print $2; exit}' " . file)
batchsize   = system("awk -F'INFO: ' '/Batch size:/{print $2; exit}' " . file)
learnrate   = system("awk -F'INFO: ' '/Learning rate:/{print $2; exit}' " . file)
lossweights = system("awk -F'INFO: ' '/WeightedEnergyForcesLoss/{print $2; exit}' " . file)
heads       = system("awk -F'INFO: ' '/Using heads:/{print $2; exit}' " . file)
elements    = system("awk -F'INFO: ' '/Atomic Numbers used:/{print $2; exit}' " . file)

last_epoch = system("awk -F= '/^last_epoch=/{print $2}' " . metafile)
last_loss  = system("awk -F= '/^last_loss=/{print $2}' " . metafile)
last_e     = system("awk -F= '/^last_e=/{print $2}' " . metafile)
last_f     = system("awk -F= '/^last_f=/{print $2}' " . metafile)
init_loss  = system("awk -F= '/^init_loss=/{print $2}' " . metafile)
init_e     = system("awk -F= '/^init_e=/{print $2}' " . metafile)
init_f     = system("awk -F= '/^init_f=/{print $2}' " . metafile)
best_loss  = system("awk -F= '/^best_loss=/{print $2}' " . metafile)
best_e     = system("awk -F= '/^best_e=/{print $2}' " . metafile)
best_f     = system("awk -F= '/^best_f=/{print $2}' " . metafile)

# -------------------------
# Window control
# -------------------------
xmax = 0
if (strlen(last_epoch) > 0) xmax = int(last_epoch)

if (xmax < win) {
    xmin = -1
    xmax_plot = win
} else {
    xmin = xmax - win + 1
    xmax_plot = xmax
}

# -------------------------
# Colors / styles
# -------------------------
loss_color = "#00d7ff"   # cyan
f_color    = "#ffd700"   # yellow/gold
e_color    = "#00ff87"   # green
trade_color= "#ff5fff"   # magenta

# -------------------------
# Global layout
# -------------------------
set multiplot title sprintf("MACE Training Monitor (window=%d epochs, refresh=%ds)", int(win), int(refresh))

unset label
set label 1  sprintf("file: %s", fname) at screen 0.02,0.975 left
set label 2  sprintf("heads: %s", heads) at screen 0.38,0.975 left
set label 3  sprintf("elements: %s", elements) at screen 0.68,0.975 left

set label 4  sprintf("foundation: %s", foundation) at screen 0.02,0.94 left
set label 5  sprintf("%s", train_valid) at screen 0.02,0.905 left
set label 6  sprintf("neighbors: %s", neighbors) at screen 0.02,0.87 left
set label 7  sprintf("%s", batchsize) at screen 0.40,0.87 left
set label 8  sprintf("%s", learnrate) at screen 0.62,0.87 left
set label 9  sprintf("%s", lossweights) at screen 0.02,0.835 left

set label 10 sprintf("last epoch=%s  loss=%s  E=%s meV  F=%s meV/A", last_epoch, last_loss, last_e, last_f) at screen 0.02,0.80 left
set label 11 sprintf("init: loss=%s  E=%s  F=%s", init_loss, init_e, init_f) at screen 0.02,0.765 left
set label 12 sprintf("best: loss=%s  E=%s  F=%s", best_loss, best_e, best_f) at screen 0.42,0.765 left

set grid
set key off
set lmargin 8
set rmargin 3
set tmargin 2
set bmargin 2

# =========================
# Manual panel placement
# =========================

# -------------------------
# Panel 1: Validation loss
# -------------------------
set origin 0.05, 0.42
set size   0.40, 0.26
set title "Validation loss"
set xlabel "Epoch"
set ylabel "Loss"
set xrange [xmin:xmax_plot]
plot datafile using 2:3 with lines lw 2 lc rgb loss_color

# -------------------------
# Panel 2: Validation RMSE_F
# -------------------------
set origin 0.57, 0.42
set size   0.38, 0.26
set title "Validation RMSE_F"
set xlabel "Epoch"
set ylabel "meV / A"
set xrange [xmin:xmax_plot]
plot datafile using 2:5 with lines lw 2 lc rgb f_color

# -------------------------
# Panel 3: Validation RMSE_E_per_atom
# -------------------------
set origin 0.05, 0.10
set size   0.40, 0.26
set title "Validation RMSE_E_per_atom"
set xlabel "Epoch"
set ylabel "meV / atom"
set xrange [xmin:xmax_plot]
plot datafile using 2:4 with lines lw 2 lc rgb e_color

# -------------------------
# Panel 4: E vs F tradeoff
# -------------------------
set origin 0.57, 0.10
set size   0.38, 0.26
set title "RMSE_E vs RMSE_F"
set xlabel "RMSE_E (meV / atom)"
set ylabel "RMSE_F (meV / A)"
unset xrange
plot datafile using 4:5 with points pt 7 ps 1.2 lc rgb trade_color

unset multiplot
pause int(refresh)
reread