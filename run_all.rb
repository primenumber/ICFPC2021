require 'parallel'
require 'net/http'

$ENDPOINT = 'https://poses.live'
$TOKEN = File::open('TOKEN.txt').read
$PATH = "target/release/icfpc2021"

def post_answer(id)
  data = File::open("best/#{id}.out").read
  url = URI.parse("#{$ENDPOINT}/api/problems/#{id}/solutions")
  http = Net::HTTP.new(url.host, url.port)
  http.use_ssl = true
  response = http.post(url.request_uri, data, header = { 'Authorization': "Bearer #{$TOKEN}" })
  response.body
end

def post_all
  (1..59).each {|id|
    puts id
    puts "ID= #{post_answer(id)}"
  }
end

def run_solve
  results = Parallel.map(1..59) {|i|
    score = `#{$PATH} solve -p problems/#{i}.in -a answer/#{i}.out`.to_i
    puts "##{i}: #{score}"
    old_score = `#{$PATH} score -p problems/#{i}.in -a best/#{i}.out`.to_i
    if score < old_score then
      puts "Updated! #{i}"
      `cp answer/#{i}.out best/#{i}.out`
      post_answer(i)
    end
    score
  }
end

run_solve
