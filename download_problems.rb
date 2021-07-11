require 'net/http'

$ENDPOINT = 'https://poses.live'
$TOKEN = File::open('TOKEN.txt').read

def get_problem(id)
  url = URI.parse("#{$ENDPOINT}/api/problems/#{id}")
  http = Net::HTTP.new(url.host, url.port)
  http.use_ssl = true
  response = http.get(url.request_uri, header = { 'Authorization': "Bearer #{$TOKEN}" })
  response.body
end

(1..106).each {|id|
  puts id
  File.open("problems/#{id}.in", mode="w") {|f| f.write(get_problem(id))}
}
