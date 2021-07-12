max_scores = File::open("max_scores.txt").readlines
min_costs = File::open("min_costs.txt").readlines
normal_costs = File::open("best_07121804.txt").readlines
globalist_costs = File::open("best_globalist.txt").readlines
list = 132.times.map {|i|
  mxs = max_scores[i].split[1].to_i
  nrc = normal_costs[i].split[1].to_i
  glc = globalist_costs[i].split[1].to_i
  mnc = [min_costs[i].to_i, glc].min
  estimated = mxs * Math.sqrt(mnc + 1) * (1.0 / Math.sqrt(glc + 1.0) - 1.0 / Math.sqrt(nrc + 1.0))
  [estimated, i+1]
}
list.sort!
list.each {|s,i|
  puts "#{i} #{s}"
}
