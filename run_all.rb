require 'parallel'
require 'net/http'
require 'securerandom'

$ENDPOINT = 'https://poses.live'
$TOKEN = File::open('TOKEN.txt').read
$PATH = "target/release/icfpc2021"

$RANGE = 1..132

def touch(id)
  `touch flags/#{id}.flag`
end

def post_answer(id)
  path = calculate_best(id)[1]
  return if path == nil
  f = File.open(path)
  return if f.mtime <= File::mtime("flags/#{id}.flag")
  puts "Updated: #{id}"
  data = f.read
  url = URI.parse("#{$ENDPOINT}/api/problems/#{id}/solutions")
  http = Net::HTTP.new(url.host, url.port)
  http.use_ssl = true
  response = http.post(url.request_uri, data, header = { 'Authorization': "Bearer #{$TOKEN}" })
  touch(id)
  response.body
end

def post_all
  $RANGE.each {|id|
    puts id
    puts "ID= #{post_answer(id)}"
  }
end

def run_solve_impl(i, loop_count, quiet=false)
  id = SecureRandom.uuid
  score = `#{$PATH} solve -p problems/#{i}.in -a answer/#{i}_#{id}.out -n #{loop_count}`.to_i
  if !quiet then
    puts "##{i}: #{score}"
  end
  score
end

def calculate_best(i)
  best_score = 1e18
  best_path = nil
  Dir.glob("#{i}_*.out", base: "answer") { |path|
    score = `#{$PATH} score -p problems/#{i}.in -a answer/#{path}`.to_i
    if score < best_score then
      best_score = score
      best_path = "answer/" + path
    end
  }
  [best_score, best_path]
end

def run_solve(loop_count)
  Parallel.each($RANGE.cycle(10)) {|i|
    score = run_solve_impl(i, loop_count, true)
  }
end

def run_solve_remote(loop_count)
  nodes = 4
  Parallel.each(0...nodes) {|nid|
    name = sprintf("aries%02x", nid)
    `scp run_all.rb #{name}:ICFPC2021/`
    `scp #{$PATH} #{name}:ICFPC2021/target/release/`
    `ssh #{name} '. .bashrc; cd ICFPC2021; bundle exec ruby run_all.rb solve'`
    #`scp #{name}:ICFPC2021/answer/* answer/`
    `rsync -au aries03:ICFPC2021/answer/ answer/`
  }
end

case ARGV[0]
when "solve" then
  if ARGV[1] == nil then
    run_solve(20000000)
  elsif ARGV[1] == "remote" then
    run_solve_remote(20000000)
  else
    run_solve_impl(ARGV[1].to_i, 20000000)
  end
when "post" then
  if ARGV[1] == nil then
    Parallel.each($RANGE) {|i|
      puts i, post_answer(i)
    }
  else
    puts post_answer(ARGV[1].to_i)
  end
when "best" then
  Parallel.map($RANGE) {|i|
    score, path = calculate_best(i)
    if path != nil then
      "#{i} #{score} #{path}"
    else
      "#{i} Not solved"
    end
  }.each {|s|
    puts s
  }
when "touch" then
  $RANGE.each {|i|
    touch(i)
  }
end
