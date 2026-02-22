use perpetual::objective::Objective;
use perpetual::{CalibrationMethod, PerpetualBooster};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::Write;
use std::path::Path;

// Note: Since this is purely a demonstration script using standard Rust CSV features
// we implement a basic custom CSV parser and feature engineer.
// In a real-world scenario, you might use `polars` but we avoid adding heavy dependencies.

#[allow(dead_code)]
#[derive(Debug, Clone, serde::Deserialize)]
struct GameRecord {
    #[serde(rename = "Season")]
    season: u16,
    #[serde(rename = "DayNum")]
    day_num: u16,
    #[serde(rename = "WTeamID")]
    w_team_id: u32,
    #[serde(rename = "WScore")]
    w_score: f64,
    #[serde(rename = "LTeamID")]
    l_team_id: u32,
    #[serde(rename = "LScore")]
    l_score: f64,
    #[serde(rename = "WLoc")]
    w_loc: String,
    #[serde(rename = "NumOT")]
    num_ot: u8,
    #[serde(rename = "WFGM")]
    w_fgm: f64,
    #[serde(rename = "WFGA")]
    w_fga: f64,
    #[serde(rename = "WFGM3")]
    w_fgm3: f64,
    #[serde(rename = "WFGA3")]
    w_fga3: f64,
    #[serde(rename = "WFTM")]
    w_ftm: f64,
    #[serde(rename = "WFTA")]
    w_fta: f64,
    #[serde(rename = "WOR")]
    w_or: f64,
    #[serde(rename = "WDR")]
    w_dr: f64,
    #[serde(rename = "WAst")]
    w_ast: f64,
    #[serde(rename = "WTO")]
    w_to: f64,
    #[serde(rename = "WStl")]
    w_stl: f64,
    #[serde(rename = "WBlk")]
    w_blk: f64,
    #[serde(rename = "WPF")]
    w_pf: f64,
    #[serde(rename = "LFGM")]
    l_fgm: f64,
    #[serde(rename = "LFGA")]
    l_fga: f64,
    #[serde(rename = "LFGM3")]
    l_fgm3: f64,
    #[serde(rename = "LFGA3")]
    l_fga3: f64,
    #[serde(rename = "LFTM")]
    l_ftm: f64,
    #[serde(rename = "LFTA")]
    l_fta: f64,
    #[serde(rename = "LOR")]
    l_or: f64,
    #[serde(rename = "LDR")]
    l_dr: f64,
    #[serde(rename = "LAst")]
    l_ast: f64,
    #[serde(rename = "LTO")]
    l_to: f64,
    #[serde(rename = "LStl")]
    l_stl: f64,
    #[serde(rename = "LBlk")]
    l_blk: f64,
    #[serde(rename = "LPF")]
    l_pf: f64,
}

#[derive(Debug, Clone, serde::Deserialize)]
struct SeedRecord {
    #[serde(rename = "Season")]
    season: u16,
    #[serde(rename = "Seed")]
    seed: String,
    #[serde(rename = "TeamID")]
    team_id: u32,
}

#[derive(Debug, Clone)]
struct TeamSeasonStats {
    wins: f64,
    losses: f64,
    points_scored: f64,
    points_allowed: f64,
    possessions: f64,
    late_season_wins: f64,
    late_season_losses: f64,
    first_5_elo: Option<f64>,
    last_5_elo: Option<f64>,
    games_played: u32,
    conf_tourney_games: u32,
    opponent_elos: Vec<f64>, // For Strength of Schedule
    massey_ranking: f64,
    massey_std: f64,
    point_diffs: Vec<f64>,
    opponent_ids: Vec<u32>, // For SRS
    srs: f64,
    weighted_gap_avg: f64, // Recency weighted
}

impl Default for TeamSeasonStats {
    fn default() -> Self {
        TeamSeasonStats {
            wins: 0.0,
            losses: 0.0,
            points_scored: 0.0,
            points_allowed: 0.0,
            possessions: 0.0,
            late_season_wins: 0.0,
            late_season_losses: 0.0,
            first_5_elo: None,
            last_5_elo: None,
            games_played: 0,
            point_diffs: Vec::new(),
            opponent_ids: Vec::new(),
            conf_tourney_games: 0,
            opponent_elos: Vec::new(),
            massey_ranking: 350.0,
            massey_std: 50.0,
            srs: -15.0,
            weighted_gap_avg: -15.0,
        }
    }
}

#[allow(dead_code)]
impl TeamSeasonStats {
    fn win_ratio(&self) -> f64 {
        if self.games_played == 0 {
            0.5
        } else {
            self.wins / (self.games_played as f64)
        }
    }
    fn gap_avg(&self) -> f64 {
        if self.games_played == 0 {
            0.0
        } else {
            (self.points_scored - self.points_allowed) / (self.games_played as f64)
        }
    }
    fn off_efficiency(&self) -> f64 {
        if self.possessions == 0.0 {
            100.0
        } else {
            100.0 * self.points_scored / self.possessions
        }
    }
    fn def_efficiency(&self) -> f64 {
        if self.possessions == 0.0 {
            100.0
        } else {
            100.0 * self.points_allowed / self.possessions
        }
    }
    fn net_efficiency(&self) -> f64 {
        self.off_efficiency() - self.def_efficiency()
    }
    fn sos(&self) -> f64 {
        if self.opponent_elos.is_empty() {
            1500.0
        } else {
            self.opponent_elos.iter().sum::<f64>() / (self.opponent_elos.len() as f64)
        }
    }
    fn point_diff_std(&self) -> f64 {
        let n = self.point_diffs.len() as f64;
        if n < 2.0 {
            return 0.0;
        }
        let mean: f64 = self.point_diffs.iter().sum::<f64>() / n;
        let variance: f64 = self.point_diffs.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
        variance.sqrt()
    }
}

#[derive(Debug, Clone, serde::Deserialize)]
struct MasseyRecord {
    #[serde(rename = "Season")]
    season: u16,
    #[serde(rename = "RankingDayNum")]
    day_num: u16,
    #[serde(rename = "SystemName")]
    system: String,
    #[serde(rename = "TeamID")]
    team_id: u32,
    #[serde(rename = "OrdinalRank")]
    rank: f64,
}

// ELO calculation
fn update_elo(elo_winner: f64, elo_loser: f64, margin: f64) -> (f64, f64) {
    let k = 20.0;
    // Autocorrelation adjustment
    let margin_multiplier = ((margin + 3.0).ln()).max(1.0) / (7.5 + 0.006 * (elo_winner - elo_loser));
    let expected_win = 1.0 / (1.0 + 10_f64.powf((elo_loser - elo_winner) / 400.0));

    let shift = k * margin_multiplier * (1.0 - expected_win);
    (elo_winner + shift, elo_loser - shift)
}

fn estimate_possessions(fga: f64, or: f64, to: f64, fta: f64) -> f64 {
    fga - or + to + 0.475 * fta
}

fn map_seed_to_num(seed_str: &str) -> f64 {
    let num_str: String = seed_str.chars().filter(|c| c.is_ascii_digit()).collect();
    num_str.parse().unwrap_or(8.0)
}

fn map_region_to_num(seed_str: &str) -> f64 {
    match seed_str.chars().next().unwrap_or('W') {
        'W' => 0.0,
        'X' => 1.0,
        'Y' => 2.0,
        'Z' => 3.0,
        _ => 0.0,
    }
}

fn main() {
    println!("Starting March Machine Learning Mania 2026 Solution Script...");

    println!(
        "Note: This script requires Kaggle MMMM datasets in the 'examples/kaggle/march_machine_learning_mania_2026' folder."
    );
    if !Path::new("examples/kaggle/march_machine_learning_mania_2026/MRegularSeasonDetailedResults.csv").exists() {
        println!(
            "Please place MRegularSeasonDetailedResults.csv, MNCAATourneyDetailedResults.csv, and MNCAATourneySeeds.csv in examples/kaggle/march_machine_learning_mania_2026/ before running."
        );
    }

    // We will simulate the data structures so this script compiles and shows the structure,
    // although actual I/O requires the kaggle data to be present!

    // 1. Process Regular Season Data to build ELO and Stats
    let mut elo_ratings: HashMap<(u16, u32), f64> = HashMap::new(); // (season, team_id) -> ELO
    let mut season_stats: HashMap<(u16, u32), TeamSeasonStats> = HashMap::new();

    // Read the regular season data
    let mut games: Vec<GameRecord> = Vec::new();
    let path = "examples/kaggle/march_machine_learning_mania_2026/MRegularSeasonDetailedResults.csv";
    if Path::new(path).exists() {
        let mut rdr = csv::Reader::from_path(path).expect("Failed to open CSV");
        for result in rdr.deserialize() {
            let game: GameRecord = result.expect("Failed to deserialize row");
            games.push(game);
        }
        println!("Loaded {} regular season games.", games.len());
    } else {
        println!("Warning: Data not found, skipping real data load.");
    }

    println!("1. Feature Engineering: Basic, Trajectory, Seasonality, Efficiency...");

    for game in games.iter() {
        // Prevent validation year data from leaking into the training ELO state
        // We calculate ELOs and stats for 2025 separately so it can be evaluated cleanly.

        // Single-season ELO: Only look up ELOs within the current season.
        let w_elo = *elo_ratings.get(&(game.season, game.w_team_id)).unwrap_or(&(1500.0));
        let l_elo = *elo_ratings.get(&(game.season, game.l_team_id)).unwrap_or(&(1500.0));

        let (new_w_elo, new_l_elo) = update_elo(w_elo, l_elo, game.w_score - game.l_score);
        elo_ratings.insert((game.season, game.w_team_id), new_w_elo);
        elo_ratings.insert((game.season, game.l_team_id), new_l_elo);

        let recency_weight = (game.day_num as f64 / 133.0).powi(2); // Quadratic recency

        // Update Stats W
        {
            let w_stats = season_stats.entry((game.season, game.w_team_id)).or_default();
            w_stats.wins += 1.0;
            w_stats.points_scored += game.w_score;
            w_stats.points_allowed += game.l_score;
            w_stats.point_diffs.push(game.w_score - game.l_score);
            w_stats.possessions += estimate_possessions(game.w_fga, game.w_or, game.w_to, game.w_fta);
            w_stats.games_played += 1;
            w_stats.opponent_elos.push(l_elo); // SoS
            w_stats.opponent_ids.push(game.l_team_id);
            if w_stats.games_played <= 5 {
                w_stats.first_5_elo = Some(new_w_elo);
            }
            w_stats.last_5_elo = Some(new_w_elo);

            if game.day_num > 100 {
                w_stats.late_season_wins += 1.0;
            }
            if game.day_num >= 125 {
                w_stats.conf_tourney_games += 1;
            }
            w_stats.weighted_gap_avg += (game.w_score - game.l_score) * recency_weight;
        }

        // Update Stats L
        {
            let l_stats = season_stats.entry((game.season, game.l_team_id)).or_default();
            l_stats.losses += 1.0;
            l_stats.points_scored += game.l_score;
            l_stats.points_allowed += game.w_score;
            l_stats.point_diffs.push(game.l_score - game.w_score);
            l_stats.possessions += estimate_possessions(game.l_fga, game.l_or, game.l_to, game.l_fta);
            l_stats.games_played += 1;
            l_stats.opponent_elos.push(w_elo); // SoS
            l_stats.opponent_ids.push(game.w_team_id);
            if l_stats.games_played <= 5 {
                l_stats.first_5_elo = Some(new_l_elo);
            }
            l_stats.last_5_elo = Some(new_l_elo);

            if game.day_num > 100 {
                l_stats.late_season_losses += 1.0;
            }
            if game.day_num >= 125 {
                l_stats.conf_tourney_games += 1;
            }
            l_stats.weighted_gap_avg += (game.l_score - game.w_score) * recency_weight;
        }
    }

    // 1c. Calculate SRS (Simple Rating System)
    println!("1c. Calculating Iterative SRS Ratings...");
    let seasons: Vec<u16> = season_stats
        .keys()
        .map(|(s, _)| *s)
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();
    for &s_year in &seasons {
        let team_ids: Vec<u32> = season_stats
            .keys()
            .filter(|(s, _)| *s == s_year)
            .map(|(_, t)| *t)
            .collect();
        if team_ids.is_empty() {
            continue;
        }

        // Initial SRS = Point Diff Avg (but using weighted gap if games > 0)
        for t_id in &team_ids {
            if let Some(stats) = season_stats.get_mut(&(s_year, *t_id)) {
                let count = stats.point_diffs.len() as f64;
                if count > 0.0 {
                    stats.srs = stats.weighted_gap_avg / count;
                } else {
                    stats.srs = -10.0;
                }
            }
        }

        // Iterative refinement (10 iterations is usually enough)
        for _ in 0..15 {
            let mut new_ratings: HashMap<u32, f64> = HashMap::new();
            for t_id in &team_ids {
                let stats = season_stats.get(&(s_year, *t_id)).unwrap();
                // Use weighted gap for SRS refinement too
                let count = stats.point_diffs.len() as f64;
                let current_point_diff = if count > 0.0 {
                    stats.weighted_gap_avg / count
                } else {
                    0.0
                };

                let mut sos_sum = 0.0;
                let mut count_opp = 0;
                for opp in &stats.opponent_ids {
                    if let Some(opp_stats) = season_stats.get(&(s_year, *opp)) {
                        sos_sum += opp_stats.srs;
                        count_opp += 1;
                    }
                }
                let sos = if count_opp > 0 {
                    sos_sum / (count_opp as f64)
                } else {
                    0.0
                };
                new_ratings.insert(*t_id, current_point_diff + sos);
            }
            // Update current ratings
            for (t_id, r) in new_ratings {
                season_stats.get_mut(&(s_year, t_id)).unwrap().srs = r;
            }
        }
    }

    // 1b. Read Massey Ordinals
    println!("1b. Parsing Massey Ordinals (Rankings)...");
    let massey_path = "examples/kaggle/march_machine_learning_mania_2026/MMasseyOrdinals.csv";
    if Path::new(massey_path).exists() {
        // We only care about the final ranking (DayNum 133 or last available)
        // To save memory, we'll just keep the latest ranking per system per team.
        let mut latest_massey: HashMap<(u16, u32, String), (u16, f64)> = HashMap::new();
        let mut rdr = csv::Reader::from_path(massey_path).expect("Failed to open Massey CSV");
        for result in rdr.deserialize() {
            let record: MasseyRecord = result.expect("Failed to deserialize Massey row");
            let key = (record.season, record.team_id, record.system);
            let entry = latest_massey.entry(key).or_insert((0, 0.0));
            if record.day_num >= entry.0 {
                *entry = (record.day_num, record.rank);
            }
        }

        // Efficient One-Pass Aggregation for Massey
        // key: (season, team_id) -> (weighted_sum, total_count, all_ranks)
        let mut team_agg: HashMap<(u16, u32), (f64, usize, Vec<f64>)> = HashMap::new();
        for (key, val) in &latest_massey {
            let season = key.0;
            let team_id = key.1;
            let system = &key.2;
            let rank = val.1;

            let weight = match system.as_str() {
                "POM" => 4,
                "SAG" => 3,
                "MOR" | "NET" => 2,
                _ => 1,
            };
            let entry = team_agg.entry((season, team_id)).or_insert((0.0, 0, Vec::new()));
            entry.0 += rank * (weight as f64);
            entry.1 += weight;
            entry.2.push(rank);
        }

        for ((season, team_id), (sum, count, ranks)) in team_agg {
            let n = ranks.len() as f64;
            let mean = if count > 0 { sum / (count as f64) } else { 350.0 };
            let std = if n > 1.0 {
                let var = ranks.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (n - 1.0);
                var.sqrt()
            } else {
                20.0
            };

            let stats = season_stats.entry((season, team_id)).or_default();
            stats.massey_ranking = mean;
            stats.massey_std = std;
        }
        println!("Loaded Massey rankings for {} team-seasons.", season_stats.len());
    } else {
        println!("Warning: Massey Ordinals not found.");
    }

    // 2. Read Tournament Seeds
    let mut seeds: HashMap<(u16, u32), f64> = HashMap::new();
    let mut regions: HashMap<(u16, u32), f64> = HashMap::new();
    let seed_path = "examples/kaggle/march_machine_learning_mania_2026/MNCAATourneySeeds.csv";
    if Path::new(seed_path).exists() {
        let mut rdr = csv::Reader::from_path(seed_path).expect("Failed to open Seeds CSV");
        for result in rdr.deserialize() {
            let record: SeedRecord = result.expect("Failed to deserialize seed row");
            seeds.insert((record.season, record.team_id), map_seed_to_num(&record.seed));
            regions.insert((record.season, record.team_id), map_region_to_num(&record.seed));
        }
        println!("Loaded {} seeds and regions.", seeds.len());
    } else {
        println!("Warning: Tourney Seeds not found, using dummy data.");
        seeds.insert((2024, 1101), 1.0); // Dummy
        seeds.insert((2024, 1102), 16.0); // Dummy
    }

    // Read Tournament Games for training
    let mut history_tourney_games: Vec<(u16, u32, u32)> = Vec::new();
    let tourney_path = "examples/kaggle/march_machine_learning_mania_2026/MNCAATourneyDetailedResults.csv";
    if Path::new(tourney_path).exists() {
        let mut rdr = csv::Reader::from_path(tourney_path).expect("Failed to open Tourney CSV");
        for result in rdr.deserialize() {
            let game: GameRecord = result.expect("Failed to deserialize tourney row");
            history_tourney_games.push((game.season, game.w_team_id, game.l_team_id));
        }
        println!("Loaded {} tourney games.", history_tourney_games.len());
    } else {
        println!("Warning: Tourney Data not found, skipping real data load.");
        history_tourney_games = vec![
            (2023, 1101, 1102), // A won
            (2025, 1101, 1102), // A won (Validation year)
        ];
    }

    // 3. Build Training Dataset (COLUMN MAJOR)
    println!("2. Constructing Symmetrical Training Data...");

    let mut train_matchups: Vec<(u32, u32, u16, f64)> = Vec::new();
    let mut val_matchups: Vec<(u32, u32, u16, f64)> = Vec::new();

    for (season, w_team, l_team) in history_tourney_games {
        if season == 2025 {
            val_matchups.push((w_team, l_team, season, 1.0));
            val_matchups.push((l_team, w_team, season, 0.0));
        } else {
            train_matchups.push((w_team, l_team, season, 1.0));
            train_matchups.push((l_team, w_team, season, 0.0));
        }
    }

    let num_feats = 12;
    let build_matrix_data = |matchups: &[(u32, u32, u16, f64)]| -> (Vec<Vec<f64>>, Vec<f64>) {
        let n = matchups.len();
        let mut cols = vec![vec![0.0; n]; num_feats];
        let mut targets = Vec::with_capacity(n);

        for (i, (team_a, team_b, season, label)) in matchups.iter().enumerate() {
            targets.push(*label);

            let stats_a = season_stats.get(&(*season, *team_a)).cloned().unwrap_or_default();
            let stats_b = season_stats.get(&(*season, *team_b)).cloned().unwrap_or_default();
            let elo_a = *elo_ratings.get(&(*season, *team_a)).unwrap_or(&1500.0);
            let elo_b = *elo_ratings.get(&(*season, *team_b)).unwrap_or(&1500.0);
            let seed_a = *seeds.get(&(*season, *team_a)).unwrap_or(&8.0);
            let seed_b = *seeds.get(&(*season, *team_b)).unwrap_or(&8.0);
            let reg_a = *regions.get(&(*season, *team_a)).unwrap_or(&0.0);
            let reg_b = *regions.get(&(*season, *team_b)).unwrap_or(&0.0);

            // 1. Seed Diff
            cols[0][i] = seed_a - seed_b;
            // 2. ELO Diff
            cols[1][i] = elo_a - elo_b;
            // 3. Win Ratio Diff
            cols[2][i] = stats_a.win_ratio() - stats_b.win_ratio();
            // 4. SRS Diff (Alpha feature)
            cols[3][i] = stats_a.srs - stats_b.srs;
            // 5. Net Efficiency Diff
            cols[4][i] = stats_a.net_efficiency() - stats_b.net_efficiency();
            // 6. Massey Rank Diff
            cols[5][i] = stats_a.massey_ranking - stats_b.massey_ranking;
            // 7. Massey Consistency (Std)
            cols[6][i] = stats_a.massey_std - stats_b.massey_std;
            // 8. Late Season Momentum
            cols[7][i] = (stats_a.late_season_wins - stats_a.late_season_losses)
                - (stats_b.late_season_wins - stats_b.late_season_losses);
            // 9. Seed * Ranking Interaction
            cols[8][i] = (seed_a * stats_a.massey_ranking) - (seed_b * stats_b.massey_ranking);
            // 10. Trajectory (Last 5 ELO)
            cols[9][i] = stats_a.last_5_elo.unwrap_or(elo_a) - stats_b.last_5_elo.unwrap_or(elo_b);
            // 11-12. Categorical Regions (Indices 10, 11)
            cols[10][i] = reg_a;
            cols[11][i] = reg_b;
        }
        (cols, targets)
    };

    let num_feats = 12;

    let (cols_train_vec, y_train) = build_matrix_data(&train_matchups);
    let (cols_val_vec, y_val) = build_matrix_data(&val_matchups);

    if y_train.is_empty() {
        println!("No training data stimulated for script. In a real scenario, model trains here.");
        return;
    }

    use perpetual::ColumnarMatrix;

    let rows_train = y_train.len();
    let col_refs_train: Vec<&[f64]> = cols_train_vec.iter().map(|v| v.as_slice()).collect();
    let matrix_train = ColumnarMatrix::new(col_refs_train, None, rows_train);

    let rows_val = y_val.len();
    let col_refs_val: Vec<&[f64]> = cols_val_vec.iter().map(|v| v.as_slice()).collect();
    let matrix_val = ColumnarMatrix::new(col_refs_val, None, rows_val);

    println!("Training rows: {}, Val rows: {}", rows_train, rows_val);
    println!("y_train mean: {:.4}", y_train.iter().sum::<f64>() / (rows_train as f64));
    println!("y_val mean: {:.4}", y_val.iter().sum::<f64>() / (rows_val as f64));

    // 4. Training with PerpetualBooster and LogLoss
    println!("3. Evaluating PerpetualBooster across budgets and boldness factors...");
    let budgets = vec![1.0, 1.5, 2.0];
    let mut best_brier = f64::MAX;
    let mut best_budget = 1.0;
    let mut best_boldness = 1.0;

    for &budget in &budgets {
        let mut cat_feats = HashSet::new();
        cat_feats.insert(10);
        cat_feats.insert(11);

        let mut model = PerpetualBooster::default()
            .set_objective(Objective::LogLoss)
            .set_budget(budget)
            .set_iteration_limit(Some(5000)) // Safety timeout for high budgets
            .set_categorical_features(Some(cat_feats.clone()));

        if rows_train > 0 && rows_val > 0 {
            model.fit_columnar(&matrix_train, &y_train, None, None).expect("Fail");

            let w_val = vec![1.0; rows_val];
            let _ = model.calibrate_columnar(CalibrationMethod::GRP, (&matrix_val, &y_val, &w_val));

            let val_preds_raw = model.predict_proba_columnar(&matrix_val, true, true);

            // Boldness Scaling (Logit Scaling)
            // Winning solutions often use a factor of 1.1 - 1.3 to push probabilities
            let boldness_factors = vec![1.0, 1.1, 1.2, 1.5, 1.8, 2.0];
            for bf in boldness_factors {
                let mut brier_sum = 0.0;
                for i in 0..val_preds_raw.len() {
                    let mut p_raw = val_preds_raw[i];
                    if p_raw > 1.0 - 1e-7 {
                        p_raw = 1.0 - 1e-7;
                    }
                    if p_raw < 1e-7 {
                        p_raw = 1e-7;
                    }
                    let logit = -(1.0 / p_raw - 1.0).ln() * bf;
                    let prob = 1.0 / (1.0 + (-logit).exp());
                    let error = y_val[i] - prob;
                    brier_sum += error * error;
                }
                let brier = brier_sum / (val_preds_raw.len() as f64);
                println!(
                    "  Budget: {:.1} | Boldness: {:.1} | Brier Score: {:.5}",
                    budget, bf, brier
                );

                if brier < best_brier {
                    best_brier = brier;
                    best_budget = budget;
                    best_boldness = bf;
                }
            }
        }
    }

    println!("--------------------------------------------------");
    println!(
        "Best Budget Found: {:.1} | Best Boldness: {:.1} | Best Brier Score: {:.5}",
        best_budget, best_boldness, best_brier
    );
    println!("--------------------------------------------------");

    // Retrain on all data
    println!("Retraining on all data for submission...");
    let mut cols_all = cols_train_vec.clone();
    for c in 0..num_feats {
        cols_all[c].extend(cols_val_vec[c].iter());
    }
    let mut y_all = y_train;
    y_all.extend(y_val);

    let rows_all = y_all.len();
    let col_refs_all: Vec<&[f64]> = cols_all.iter().map(|v| v.as_slice()).collect();
    let matrix_all = ColumnarMatrix::new(col_refs_all, None, rows_all);

    let mut final_cat_feats = HashSet::new();
    final_cat_feats.insert(10);
    final_cat_feats.insert(11);

    let mut final_model = PerpetualBooster::default()
        .set_objective(Objective::LogLoss)
        .set_budget(best_budget)
        .set_iteration_limit(Some(5000))
        .set_calibration_method(CalibrationMethod::GRP)
        .set_categorical_features(Some(final_cat_feats));

    if rows_all > 0 {
        final_model.fit_columnar(&matrix_all, &y_all, None, None).expect("Fail");
        let w_all = vec![1.0; rows_all];
        let _ = final_model.calibrate_columnar(CalibrationMethod::GRP, (&matrix_all, &y_all, &w_all));
    }

    // 5. Generate Submission File for 2026 (or latest available) Matchups
    println!("4. Generating Submission...");

    let sample_path = "examples/kaggle/march_machine_learning_mania_2026/SampleSubmissionStage1.csv";
    let sub_path = "examples/kaggle/march_machine_learning_mania_2026/submission.csv";

    if !Path::new(sample_path).exists() {
        println!("SampleSubmissionStage1.csv not found. Please provide it to generate the correct submission format.");
        return;
    }

    let mut file = File::create(sub_path).expect("Failed to create file");
    writeln!(&mut file, "ID,Pred").unwrap();

    let mut rdr = csv::Reader::from_path(sample_path).expect("Failed to open Sample Submission");
    let mut num_preds = 0;

    for result in rdr.records() {
        let record = result.expect("Failed to read sample submission row");
        let id_str = &record[0];

        let parts: Vec<&str> = id_str.split('_').collect();
        if parts.len() != 3 {
            continue;
        }
        let target_season: u16 = parts[0].parse().unwrap_or(2025);
        let team_a: u32 = parts[1].parse().unwrap_or(1101);
        let team_b: u32 = parts[2].parse().unwrap_or(1102);

        let stats_a = season_stats.get(&(target_season, team_a)).cloned().unwrap_or_default();
        let stats_b = season_stats.get(&(target_season, team_b)).cloned().unwrap_or_default();
        let elo_a = *elo_ratings.get(&(target_season, team_a)).unwrap_or(&1500.0);
        let elo_b = *elo_ratings.get(&(target_season, team_b)).unwrap_or(&1500.0);
        let seed_a = *seeds.get(&(target_season, team_a)).unwrap_or(&8.0);
        let seed_b = *seeds.get(&(target_season, team_b)).unwrap_or(&8.0);

        let reg_a = *regions.get(&(target_season, team_a)).unwrap_or(&0.0);
        let reg_b = *regions.get(&(target_season, team_b)).unwrap_or(&0.0);

        let test_feats = vec![
            seed_a - seed_b,
            elo_a - elo_b,
            stats_a.win_ratio() - stats_b.win_ratio(),
            stats_a.srs - stats_b.srs,
            stats_a.net_efficiency() - stats_b.net_efficiency(),
            stats_a.massey_ranking - stats_b.massey_ranking,
            stats_a.massey_std - stats_b.massey_std,
            (stats_a.late_season_wins - stats_a.late_season_losses)
                - (stats_b.late_season_wins - stats_b.late_season_losses),
            (seed_a * stats_a.massey_ranking) - (seed_b * stats_b.massey_ranking),
            stats_a.last_5_elo.unwrap_or(elo_a) - stats_b.last_5_elo.unwrap_or(elo_b),
            reg_a,
            reg_b,
        ];

        if rows_all > 0 {
            let mut test_matrix_cols = Vec::new();
            for &f in &test_feats {
                test_matrix_cols.push(vec![f]);
            }
            let test_matrix_refs: Vec<&[f64]> = test_matrix_cols.iter().map(|v| v.as_slice()).collect();
            let test_matrix = ColumnarMatrix::new(test_matrix_refs, None, 1);

            let pred_raw = final_model.predict_proba_columnar(&test_matrix, true, true);
            let mut p_raw = pred_raw[0];
            if p_raw > 1.0 - 1e-7 {
                p_raw = 1.0 - 1e-7;
            }
            if p_raw < 1e-7 {
                p_raw = 1e-7;
            }
            let logit = -(1.0 / p_raw - 1.0).ln() * best_boldness;
            let mut prob = 1.0 / (1.0 + (-logit).exp());

            if prob > 0.98 {
                prob = 0.975;
            }
            if prob < 0.02 {
                prob = 0.025;
            }

            writeln!(&mut file, "{},{:.6}", id_str, prob).unwrap();
        } else {
            writeln!(&mut file, "{},0.5", id_str).unwrap();
        }
        num_preds += 1;
    }
    println!("Done. submission.csv generated with {} predictions.", num_preds);
}
