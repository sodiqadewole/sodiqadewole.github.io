#!/usr/bin/env ruby

require "date"
require "pathname"
require "yaml"

ROOT = Pathname.new(__dir__).parent
CONTENT_RULES = {
  "_posts" => { required: %w[title date permalink] },
  "_portfolio" => { required: %w[title collection permalink], collection: "portfolio" },
  "_publications" => { required: %w[title collection category date permalink], collection: "publications" },
  "_talks" => { required: %w[title collection type date permalink], collection: "talks" },
  "_teaching" => { required: %w[title collection date permalink], collection: "teaching" }
}.freeze

def front_matter(path)
  lines = File.readlines(path, encoding: "UTF-8")
  return nil unless lines.first&.strip == "---"

  closing = lines[1..].index { |line| line.strip == "---" }
  return nil unless closing

  YAML.safe_load(lines[1, closing + 1].join, permitted_classes: [Date], aliases: true) || {}
rescue Psych::Exception => error
  raise "invalid YAML: #{error.message.lines.first.strip}"
end

def parse_date(value)
  return value if value.is_a?(Date)

  Date.parse(value.to_s.split(" - ").first)
end

errors = []
warnings = []
permalinks = {}

CONTENT_RULES.each do |directory, rule|
  Dir[ROOT.join(directory, "**", "*.{md,markdown,mkdn,mkd,html}").to_s].sort.each do |filename|
    path = Pathname.new(filename)
    relative = path.relative_path_from(ROOT).to_s

    begin
      data = front_matter(path)
      errors << "#{relative}: missing front matter" unless data
      next unless data

      rule[:required].each do |field|
        value = data[field]
        errors << "#{relative}: missing required field '#{field}'" if value.nil? || value.to_s.strip.empty?
      end

      if rule[:collection] && data["collection"] != rule[:collection]
        errors << "#{relative}: collection must be '#{rule[:collection]}'"
      end

      parse_date(data["date"]) if data["date"]

      permalink = data["permalink"].to_s
      unless permalink.empty?
        if permalinks.key?(permalink)
          errors << "#{relative}: duplicate permalink '#{permalink}' (also #{permalinks[permalink]})"
        else
          permalinks[permalink] = relative
        end
      end

      if directory == "_posts" && data["date"]
        filename_date = path.basename.to_s[0, 10]
        front_matter_date = parse_date(data["date"]).strftime("%Y-%m-%d")
        if filename_date != front_matter_date
          warnings << "#{relative}: filename date #{filename_date} differs from front matter date #{front_matter_date}"
        end
      end

      data.each_value do |value|
        if value.to_s.match?(/exampleurl\.com|yourorcidurl|paper-title-number|blog-post-1|tutorial-1/i)
          warnings << "#{relative}: contains a placeholder value"
        end
      end
    rescue ArgumentError => error
      errors << "#{relative}: invalid date (#{error.message})"
    rescue RuntimeError => error
      errors << "#{relative}: #{error.message}"
    end
  end
end

page_urls = Dir[ROOT.join("_pages", "**", "*.{md,html}").to_s].map do |filename|
  data = front_matter(Pathname.new(filename))
  data && data["permalink"].to_s
rescue RuntimeError
  nil
end.compact

navigation = YAML.safe_load(File.read(ROOT.join("_data", "navigation.yml")), aliases: true) || {}
(navigation["main"] || []).each do |item|
  url = item["url"].to_s
  page_url = url.split("#", 2).first
  errors << "_data/navigation.yml: URL '#{url}' does not match a page permalink" unless page_urls.include?(page_url)
end

puts "Content validation: #{errors.empty? ? 'passed' : 'failed'}"
warnings.each { |warning| warn "warning: #{warning}" }
errors.each { |error| warn "error: #{error}" }
exit(errors.empty? ? 0 : 1)