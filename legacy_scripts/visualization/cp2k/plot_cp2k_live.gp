if (!exists("file")) {
    print "Usage: plotcp2k <cp2k.log>"
    exit
}

if (!exists("mode")) {
    mode = "unknown"
}

set term dumb ansi 140 42

# -------------------------
# GEO_OPT MODE
# -------------------------
if (mode eq "geo") {

    set multiplot layout 2,1 title "CP2K GEO Monitor"

    fname_cmd      = "basename " . file
    latest_step_cmd= "awk \047/Informations at step =/{s=$NF} END{print (s!=\"\"?s:\"NA\")}\047 " . file
    latest_E_cmd   = "awk \047/Informations at step =/{flag=1; next} flag && /Total Energy[[:space:]]*=/{E=$NF; flag=0} END{print (E!=\"\"?E:\"NA\")}\047 " . file
    latest_gmax_cmd= "awk \047/Max\\. gradient/{g=$(NF-1)} END{print (g!=\"\"?g:\"NA\")}\047 " . file
    latest_grms_cmd= "awk \047/RMS gradient/{g=$(NF-1)} END{print (g!=\"\"?g:\"NA\")}\047 " . file
    latest_smax_cmd= "awk \047/Max\\. step size/{g=$(NF-1)} END{print (g!=\"\"?g:\"NA\")}\047 " . file
    nstep_cmd      = "awk \047/Informations at step =/{n++} END{print (n>0?n:0)}\047 " . file

    fname       = system(fname_cmd)
    latest_step = system(latest_step_cmd)
    latest_E    = system(latest_E_cmd)
    latest_gmax = system(latest_gmax_cmd)
    latest_grms = system(latest_grms_cmd)
    latest_smax = system(latest_smax_cmd)
    nsteps_str  = system(nstep_cmd)
    nsteps      = int(nsteps_str)

    xmax = (nsteps > 40) ? nsteps : 40
    xmin = (nsteps > 40) ? (nsteps - 39) : 1

    # -------- panel 1: Energy + Max gradient --------
    unset label
    unset xrange
    unset y2tics
    set xrange [xmin:xmax]
    set autoscale y
    set autoscale y2

    set label 1 sprintf("file: %s", fname)                at screen 0.02,0.98 left
    set label 2 sprintf("latest step: %s", latest_step)   at screen 0.30,0.98 left
    set label 3 sprintf("latest E: %s", latest_E)         at screen 0.52,0.98 left textcolor rgb "cyan"
    set label 4 sprintf("max grad: %s", latest_gmax)      at screen 0.78,0.98 left textcolor rgb "red"

    set xlabel ""
    set ylabel "log10(Max gradient)" textcolor rgb "red"
    set ytics textcolor rgb "red"

    set y2label "Energy (Ha)" textcolor rgb "cyan"
    set y2tics textcolor rgb "cyan"

    set grid
    set key off

    plot \
      "< awk '\
      /Informations at step =/ {s=$NF} \
      /Max\\. gradient/ {v=$(NF-1)+0; if (v>0) print s, log(v)/log(10)}' ".file using 1:2 with lines lc rgb "red" title "log10(max grad)", \
      "< awk '\
      /Informations at step =/ {s=$NF; flag=1; next} \
      flag && /Total Energy[[:space:]]*=/ {print s,$NF; flag=0}' ".file using 1:2 axes x1y2 with lines lc rgb "cyan" title "E"

    # -------- panel 2: RMS gradient + Max step size --------
    unset label
    unset xrange
    unset y2tics
    set xrange [xmin:xmax]
    set autoscale y
    set autoscale y2

    set xlabel "GEO step"
    set ylabel "log10(RMS gradient)" textcolor rgb "magenta"
    set ytics textcolor rgb "magenta"

    set y2label "Max step size" textcolor rgb "green"
    set y2tics textcolor rgb "green"

    set grid
    set key off

    set label 1 sprintf("rms grad: %s", latest_grms) at graph 0.70,0.95 left textcolor rgb "magenta"
    set label 2 sprintf("max step: %s", latest_smax) at graph 0.70,0.88 left textcolor rgb "green"

    plot \
      "< awk '\
      /Informations at step =/ {s=$NF} \
      /RMS gradient/ {v=$(NF-1)+0; if (v>0) print s, log(v)/log(10)}' ".file using 1:2 with lines lc rgb "magenta" title "log10(rms grad)", \
      "< awk '\
      /Informations at step =/ {s=$NF} \
      /Max\\. step size/ {print s,$(NF-1)}' ".file using 1:2 axes x1y2 with lines lc rgb "green" title "max step"

    unset multiplot
}

# -------------------------
# AIMD MODE
# -------------------------
else if (mode eq "md") {

    set multiplot layout 2,1 title "CP2K AIMD Monitor"

    fname_cmd      = "basename " . file
    latest_step_cmd= "awk \047/STEP NUMBER/{getline; if ($1 ~ /^[0-9]+$/) s=$1} END{print (s!=\"\"?s:\"NA\")}\047 " . file
    latest_T_cmd   = "awk \047/STEP NUMBER/{getline; if ($1 ~ /^[0-9]+$/) t=$3} END{print (t!=\"\"?t:\"NA\")}\047 " . file
    latest_E_cmd   = "awk \047/STEP NUMBER/{getline; if ($1 ~ /^[0-9]+$/) e=$6} END{print (e!=\"\"?e:\"NA\")}\047 " . file
    nstep_cmd      = "awk \047/STEP NUMBER/{n++} END{print (n>0?n:0)}\047 " . file

    fname       = system(fname_cmd)
    latest_step = system(latest_step_cmd)
    latest_T    = system(latest_T_cmd)
    latest_E    = system(latest_E_cmd)
    nsteps_str  = system(nstep_cmd)
    nsteps      = int(nsteps_str)

    xmax = (nsteps > 40) ? nsteps : 40
    xmin = (nsteps > 40) ? (nsteps - 39) : 1

    # -------- panel 1: Temperature + Total energy --------
    unset label
    unset xrange
    unset y2tics
    set xrange [xmin:xmax]
    set autoscale y
    set autoscale y2

    set label 1 sprintf("file: %s", fname)              at screen 0.02,0.98 left
    set label 2 sprintf("latest step: %s", latest_step) at screen 0.30,0.98 left
    set label 3 sprintf("latest T: %s K", latest_T)     at screen 0.55,0.98 left textcolor rgb "red"
    set label 4 sprintf("latest E: %s", latest_E)       at screen 0.80,0.98 left textcolor rgb "cyan"

    set xlabel ""
    set ylabel "Temperature (K)" textcolor rgb "red"
    set ytics textcolor rgb "red"

    set y2label "Total energy" textcolor rgb "cyan"
    set y2tics textcolor rgb "cyan"

    set grid
    set key off

    plot \
      "< awk '\
      /STEP NUMBER/ {getline; if ($1 ~ /^[0-9]+$/) {c++; print c,$3}}' ".file using 1:2 with lines lc rgb "red" title "T", \
      "< awk '\
      /STEP NUMBER/ {getline; if ($1 ~ /^[0-9]+$/) {c++; print c,$6}}' ".file using 1:2 axes x1y2 with lines lc rgb "cyan" title "E"

    # -------- panel 2: E_kin + E_pot --------
    unset label
    unset xrange
    unset y2tics
    set xrange [xmin:xmax]
    set autoscale y

    set xlabel "MD step"
    set ylabel "Energy" textcolor rgb "magenta"
    set ytics textcolor rgb "magenta"

    set grid
    set key off

    plot \
      "< awk '\
      /STEP NUMBER/ {getline; if ($1 ~ /^[0-9]+$/) {c++; print c,$4}}' ".file using 1:2 with lines lc rgb "magenta" title "Ekin", \
      "< awk '\
      /STEP NUMBER/ {getline; if ($1 ~ /^[0-9]+$/) {c++; print c,$5}}' ".file using 1:2 with lines lc rgb "green" title "Epot"

    unset multiplot
}

# -------------------------
# UNKNOWN
# -------------------------
else {
    print "Mode not recognized yet."
}