require 'nokogiri'

html = File::open('best/Problems.html').read
html_doc = Nokogiri::HTML(html)
data = html_doc.xpath('//td').map{|e| e.content}
min_scores = 132.times.map {|i| data[i*3 + 2] }
min_scores.each {|d| puts d}
