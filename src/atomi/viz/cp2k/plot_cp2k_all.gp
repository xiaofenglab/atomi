if (!exists("file")) {
    print "Usage: plotcp2kall <cp2k.log>"
    exit
}

set term dumb ansi 150 58
set multiplot layout 2,2 title "CP2K GEO_OPT Full Convergence"

# leave space at top for text header
set tmargin 6
set bmargin 3
set lmargin 10
set rmargin 5

# ---------------- basic names ----------------

fname_cmd = "basename ".file
fname = system(fname_cmd)

# infer matching input file: xxx.log -> xxx.inp
inp = file
if (strlen(inp) > 4 && substr(inp, strlen(inp)-3, 4) eq ".log") {
    inp = substr(inp, 1, strlen(inp)-4).".inp"
}

# ---------------- latest summary values from log ----------------

step_cmd   = "awk '/Informations at step =/ {s=$(NF-1)} END{print (s!=\"\"?s:\"NA\")}' ".file
energy_cmd = "awk '/Total Energy[[:space:]]*=/ {E=$NF} END{print (E!=\"\"?E:\"NA\")}' ".file
maxg_cmd   = "awk '/Max\\. gradient[[:space:]]*=/ {v=$NF} END{print (v!=\"\"?v:\"NA\")}' ".file
rmsg_cmd   = "awk '/RMS gradient[[:space:]]*=/ {v=$NF} END{print (v!=\"\"?v:\"NA\")}' ".file
maxs_cmd   = "awk '/Max\\. step size[[:space:]]*=/ {v=$NF} END{print (v!=\"\"?v:\"NA\")}' ".file
rmss_cmd   = "awk '/RMS step size[[:space:]]*=/ {v=$NF} END{print (v!=\"\"?v:\"NA\")}' ".file
trust_cmd  = "awk '/Trust radius[[:space:]]*=/ {v=$NF} END{print (v!=\"\"?v:\"NA\")}' ".file

# latest convergence YES/NO from latest block only
cmaxs_cmd = "awk '\
/Informations at step =/ {delete blk; n=0; keep=1} \
keep {blk[++n]=$0} \
END{ \
  for(i=1;i<=n;i++) if(blk[i] ~ /Convergence in step size[[:space:]]*=/) {split(blk[i],a,\"=\"); gsub(/^[ \t]+|[ \t]+$/, \"\", a[2]); print a[2]; exit} \
  print \"NA\" \
}' ".file

crmss_cmd = "awk '\
/Informations at step =/ {delete blk; n=0; keep=1} \
keep {blk[++n]=$0} \
END{ \
  for(i=1;i<=n;i++) if(blk[i] ~ /Convergence in RMS step[[:space:]]*=/) {split(blk[i],a,\"=\"); gsub(/^[ \t]+|[ \t]+$/, \"\", a[2]); print a[2]; exit} \
  print \"NA\" \
}' ".file

cmaxg_cmd = "awk '\
/Informations at step =/ {delete blk; n=0; keep=1} \
keep {blk[++n]=$0} \
END{ \
  k=0; \
  for(i=1;i<=n;i++) if(blk[i] ~ /Conv\\. for gradients[[:space:]]*=/) { \
    k++; split(blk[i],a,\"=\"); gsub(/^[ \t]+|[ \t]+$/, \"\", a[2]); \
    if(k==1){print a[2]; exit} \
  } \
  print \"NA\" \
}' ".file

crmsg_cmd = "awk '\
/Informations at step =/ {delete blk; n=0; keep=1} \
keep {blk[++n]=$0} \
END{ \
  k=0; \
  for(i=1;i<=n;i++) if(blk[i] ~ /Conv\\. for gradients[[:space:]]*=/) { \
    k++; split(blk[i],a,\"=\"); gsub(/^[ \t]+|[ \t]+$/, \"\", a[2]); \
    if(k==2){print a[2]; exit} \
  } \
  print \"NA\" \
}' ".file

# ---------------- target limits: prefer latest log block ----------------
# max grad
target_maxg_cmd = "awk '\
/Informations at step =/ {delete blk; n=0; keep=1} \
keep {blk[++n]=$0} \
END{ \
  for(i=1;i<=n;i++) if(blk[i] ~ /Conv\\. limit for gradients[[:space:]]*=/) {split(blk[i],a,\"=\"); gsub(/^[ \t]+|[ \t]+$/, \"\", a[2]); print a[2]; exit} \
  print \"NA\" \
}' ".file

# rms grad
target_rmsg_cmd = "awk '\
/Informations at step =/ {delete blk; n=0; keep=1} \
keep {blk[++n]=$0} \
END{ \
  for(i=1;i<=n;i++) if(blk[i] ~ /Conv\\. limit for RMS grad\\.[[:space:]]*=/) {split(blk[i],a,\"=\"); gsub(/^[ \t]+|[ \t]+$/, \"\", a[2]); print a[2]; exit} \
  print \"NA\" \
}' ".file

# max step
target_maxs_cmd = "awk '\
/Informations at step =/ {delete blk; n=0; keep=1} \
keep {blk[++n]=$0} \
END{ \
  for(i=1;i<=n;i++) if(blk[i] ~ /Conv\\. limit for step size[[:space:]]*=/) {split(blk[i],a,\"=\"); gsub(/^[ \t]+|[ \t]+$/, \"\", a[2]); print a[2]; exit} \
  print \"NA\" \
}' ".file

# rms step
target_rmss_cmd = "awk '\
/Informations at step =/ {delete blk; n=0; keep=1} \
keep {blk[++n]=$0} \
END{ \
  for(i=1;i<=n;i++) if(blk[i] ~ /Conv\\. limit for RMS step[[:space:]]*=/) {split(blk[i],a,\"=\"); gsub(/^[ \t]+|[ \t]+$/, \"\", a[2]); print a[2]; exit} \
  print \"NA\" \
}' ".file

latest_step   = system(step_cmd)
latest_energy = system(energy_cmd)
latest_maxg   = system(maxg_cmd)
latest_rmsg   = system(rmsg_cmd)
latest_maxs   = system(maxs_cmd)
latest_rmss   = system(rmss_cmd)
latest_trust  = system(trust_cmd)

conv_maxs = system(cmaxs_cmd)
conv_rmss = system(crmss_cmd)
conv_maxg = system(cmaxg_cmd)
conv_rmsg = system(crmsg_cmd)

target_maxg_str = system(target_maxg_cmd)
target_rmsg_str = system(target_rmsg_cmd)
target_maxs_str = system(target_maxs_cmd)
target_rmss_str = system(target_rmss_cmd)

# ---------------- fallback to input file if missing in log ----------------

inp_maxg_cmd = "awk '\
BEGIN{g=0} \
toupper($0) ~ /&GEO_OPT/ {g=1} \
g && toupper($1)==\"MAX_FORCE\" {print $2; exit} \
g && toupper($0) ~ /&END[ \t]+GEO_OPT/ {g=0} \
' ".inp

inp_rmsg_cmd = "awk '\
BEGIN{g=0} \
toupper($0) ~ /&GEO_OPT/ {g=1} \
g && toupper($1)==\"RMS_FORCE\" {print $2; exit} \
g && toupper($0) ~ /&END[ \t]+GEO_OPT/ {g=0} \
' ".inp

inp_maxs_cmd = "awk '\
BEGIN{g=0} \
toupper($0) ~ /&GEO_OPT/ {g=1} \
g && toupper($1)==\"MAX_DR\" {print $2; exit} \
g && toupper($0) ~ /&END[ \t]+GEO_OPT/ {g=0} \
' ".inp

# no direct RMS-step keyword in input, keep common default if not in log
inp_rmss_default = "0.0015"

if (target_maxg_str eq "NA" || strlen(target_maxg_str) == 0) {
    target_maxg_str = system(inp_maxg_cmd)
}
if (target_rmsg_str eq "NA" || strlen(target_rmsg_str) == 0) {
    target_rmsg_str = system(inp_rmsg_cmd)
}
if (target_maxs_str eq "NA" || strlen(target_maxs_str) == 0) {
    target_maxs_str = system(inp_maxs_cmd)
}
if (target_rmss_str eq "NA" || strlen(target_rmss_str) == 0) {
    target_rmss_str = inp_rmss_default
}

# final hard fallbacks
if (target_maxg_str eq "NA" || strlen(target_maxg_str) == 0) { target_maxg_str = "0.001" }
if (target_rmsg_str eq "NA" || strlen(target_rmsg_str) == 0) { target_rmsg_str = "0.0007" }
if (target_maxs_str eq "NA" || strlen(target_maxs_str) == 0) { target_maxs_str = "0.002" }
if (target_rmss_str eq "NA" || strlen(target_rmss_str) == 0) { target_rmss_str = "0.0015" }

target_maxg = real(target_maxg_str)
target_rmsg = real(target_rmsg_str)
target_maxs = real(target_maxs_str)
target_rmss = real(target_rmss_str)

# ---------------- header labels ----------------

set label 1 sprintf("File: %s", fname) at screen 0.02,0.992 left
set label 2 sprintf("Latest step: %s", latest_step) at screen 0.28,0.992 left
set label 3 sprintf("Energy: %s", latest_energy) at screen 0.62,0.992 left

set label 4 sprintf("Max grad: %s", latest_maxg) at screen 0.02,0.965 left textcolor rgb "red"
set label 5 sprintf("RMS grad: %s", latest_rmsg) at screen 0.28,0.965 left textcolor rgb "blue"
set label 6 sprintf("Max step: %s", latest_maxs) at screen 0.55,0.965 left
set label 7 sprintf("RMS step: %s", latest_rmss) at screen 0.78,0.965 left

set label 8  sprintf("step size conv: %s", conv_maxs) at screen 0.02,0.938 left
set label 9  sprintf("RMS step conv: %s", conv_rmss) at screen 0.28,0.938 left
set label 10 sprintf("max grad conv: %s", conv_maxg) at screen 0.55,0.938 left
set label 11 sprintf("rms grad conv: %s", conv_rmsg) at screen 0.78,0.938 left

set label 12 sprintf("targets: maxG=%s  rmsG=%s  maxStep=%s  rmsStep=%s", \
  target_maxg_str, target_rmsg_str, target_maxs_str, target_rmss_str) \
  at screen 0.02,0.911 left

if (strlen(latest_trust) > 0 && latest_trust ne "NA") {
    set label 13 sprintf("Trust radius: %s", latest_trust) at screen 0.75,0.911 left textcolor rgb "yellow"
}

# =========================================================
# PANEL 1 : Energy vs GEO step
# =========================================================
set title "Energy convergence"
set xlabel "GEO step"
set ylabel "Energy (a.u.)"
set grid
set key off
set autoscale x
set autoscale y

plot \
"< awk '/Informations at step =/ {s=$(NF-1)} /Total Energy[[:space:]]*=/ {print s,$NF}' ".file \
using 1:2 with lines lc rgb "cyan"

# =========================================================
# PANEL 2 : Max / RMS gradient vs GEO step
# =========================================================
set title "Force convergence"
set xlabel "GEO step"
set ylabel "log10(gradient)"
set grid
set key right
set autoscale x
set yrange [-4:0]

plot \
"< awk '/Informations at step =/ {s=$(NF-1)} /Max\\. gradient[[:space:]]*=/ {v=$NF+0; if(v>0) print s,log(v)/log(10)}' ".file \
using 1:2 with lines lc rgb "red" title "Max grad", \
"< awk '/Informations at step =/ {s=$(NF-1)} /RMS gradient[[:space:]]*=/ {v=$NF+0; if(v>0) print s,log(v)/log(10)}' ".file \
using 1:2 with lines lc rgb "blue" title "RMS grad", \
log(target_maxg)/log(10) with lines lc rgb "green" dt 2 title "target max", \
log(target_rmsg)/log(10) with lines lc rgb "cyan" dt 4 title "target rms"

# =========================================================
# PANEL 3 : Step-size convergence
# =========================================================
set title "Step-size convergence"
set xlabel "GEO step"
set ylabel "log10(step size)"
set grid
set key right
set autoscale x
set yrange [-4:1]

plot \
"< awk '/Informations at step =/ {s=$(NF-1)} /Max\\. step size[[:space:]]*=/ {v=$NF+0; if(v>0) print s,log(v)/log(10)}' ".file \
using 1:2 with lines lc rgb "magenta" title "Max step", \
"< awk '/Informations at step =/ {s=$(NF-1)} /RMS step size[[:space:]]*=/ {v=$NF+0; if(v>0) print s,log(v)/log(10)}' ".file \
using 1:2 with lines lc rgb "yellow" title "RMS step", \
log(target_maxs)/log(10) with lines lc rgb "green" dt 2 title "target max step", \
log(target_rmss)/log(10) with lines lc rgb "cyan" dt 4 title "target rms step"

# =========================================================
# PANEL 4 : Energy change / trust radius
# =========================================================
set title "Energy change and trust radius"
set xlabel "GEO step"
set ylabel "dE"
set grid
set key right
set autoscale x
set autoscale y
set y2tics
set y2label "Trust radius"

if (strlen(latest_trust) > 0 && latest_trust ne "NA") {
    plot \
    "< awk '/Informations at step =/ {s=$(NF-1)} /Real energy change[[:space:]]*=/ {print s,$NF}' ".file \
    using 1:2 with lines lc rgb "red" title "Real dE", \
    "< awk '/Informations at step =/ {s=$(NF-1)} /Predicted change in energy[[:space:]]*=/ {print s,$NF}' ".file \
    using 1:2 with lines lc rgb "blue" title "Pred dE", \
    "< awk '/Informations at step =/ {s=$(NF-1)} /Trust radius[[:space:]]*=/ {print s,$NF}' ".file \
    using 1:2 axes x1y2 with lines lc rgb "yellow" title "Trust radius"
} else {
    plot \
    "< awk '/Informations at step =/ {s=$(NF-1)} /Real energy change[[:space:]]*=/ {print s,$NF}' ".file \
    using 1:2 with lines lc rgb "red" title "Real dE", \
    "< awk '/Informations at step =/ {s=$(NF-1)} /Predicted change in energy[[:space:]]*=/ {print s,$NF}' ".file \
    using 1:2 with lines lc rgb "blue" title "Pred dE"
}

unset multiplot