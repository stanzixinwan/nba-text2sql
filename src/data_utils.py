# 目标格式 (和Spider对齐)
input_text = "translate to SQL: How many teams are in the NBA? | database: nba | team: id, full_name, abbreviation, city, state, year_founded | game: ... "
target_text = "SELECT COUNT(*) FROM team"