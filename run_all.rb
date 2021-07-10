require 'parallel'
require 'net/http'

$ENDPOINT = 'https://poses.live'
$TOKEN = File::open('TOKEN.txt').read
$PATH = "target/release/icfpc2021"

$RANGE = 1..78

def post_answer(id)
  data = File::open("best/#{id}.out").read
  url = URI.parse("#{$ENDPOINT}/api/problems/#{id}/solutions")
  http = Net::HTTP.new(url.host, url.port)
  http.use_ssl = true
  response = http.post(url.request_uri, data, header = { 'Authorization': "Bearer #{$TOKEN}" })
  response.body
end

def post_all
  $RANGE.each {|id|
    puts id
    puts "ID= #{post_answer(id)}"
  }
end

def run_solve_impl(i, quiet=false)
  score = `#{$PATH} solve -p problems/#{i}.in -a answer/#{i}.out`.to_i
  if !quiet then
    puts "##{i}: #{score}"
  end
  old_score = `#{$PATH} score -p problems/#{i}.in -a best/#{i}.out`.to_i
  if score < old_score then
    puts "Updated! #{i}, Score=#{score}"
    `cp answer/#{i}.out best/#{i}.out`
    true
  else
    false
  end
end

def run_solve
  Parallel.map($RANGE) {|i|
    updated = false
    start = Process.clock_gettime(Process::CLOCK_MONOTONIC)
    loop {
      updated_new = run_solve_impl(i, true)
      updated = updated || updated_new
      now = Process.clock_gettime(Process::CLOCK_MONOTONIC)
      if now - start > 180 then  # 3min
        break
      end
      best_score = `#{$PATH} score -p problems/#{i}.in -a best/#{i}.out`.to_i
      if best_score == 0 then
        break
      end
    }
    [updated, i]
  }.each {|b,idx|
    if b
      puts idx
      post_answer(idx)
    end
  }
end

case ARGV[0]
when nil then
  run_solve
when "solve" then
  run_solve_impl(ARGV[1].to_i)
when "post" then
  puts post_answer(ARGV[1].to_i)
when "score" then
  $RANGE.each {|i|
    puts i, `#{$PATH} score -p problems/#{i}.in -a best/#{i}.out`
  }
end
