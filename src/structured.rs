use serde_json::Value;

use crate::contract::{INPUT_DIM, LEGACY_INPUT_DIM};
const NUMERIC_OFFSET: usize = LEGACY_INPUT_DIM;
const NUMERIC_SLOTS: usize = 96;
const BOOLEAN_OFFSET: usize = NUMERIC_OFFSET + NUMERIC_SLOTS;
const BOOLEAN_SLOTS: usize = 64;
const CATEGORY_OFFSET: usize = BOOLEAN_OFFSET + BOOLEAN_SLOTS;
const CATEGORY_SLOTS: usize = INPUT_DIM - CATEGORY_OFFSET;

const NUMERIC_POINTERS: [&str; 84] = [
    "/ball/drain_danger/emergency_flipper_bits",
    "/ball/flipper_approach/left/contact_frames",
    "/ball/flipper_approach/left/contact_seconds",
    "/ball/flipper_approach/left/distance_table_units",
    "/ball/flipper_approach/left/inner_tip_distance",
    "/ball/flipper_approach/left/projected_segment_fraction",
    "/ball/flipper_approach/left/toward_inner_tip_speed",
    "/ball/flipper_approach/right/contact_frames",
    "/ball/flipper_approach/right/contact_seconds",
    "/ball/flipper_approach/right/distance_table_units",
    "/ball/flipper_approach/right/inner_tip_distance",
    "/ball/flipper_approach/right/projected_segment_fraction",
    "/ball/flipper_approach/right/toward_inner_tip_speed",
    "/ball/physics/collision_resolution_state",
    "/ball/physics/screen_position/0",
    "/ball/physics/screen_position/1",
    "/ball/physics/table_position_q0/0",
    "/ball/physics/table_position_q0/1",
    "/ball/preemptive_projection/horizon_seconds",
    "/ball/preemptive_projection/linear_table_x",
    "/ball/preemptive_projection/linear_table_y",
    "/ball/preemptive_projection/velocity_to_position_scale",
    "/ball/saver_seconds",
    "/ball/speed",
    "/ball/velocity_x",
    "/ball/velocity_y",
    "/ball/x",
    "/ball/y",
    "/board/ball_catch_state",
    "/board/board_state",
    "/board/board_substate",
    "/board/catch_arrow_progress",
    "/board/catch_lights/0",
    "/board/catch_lights/1",
    "/board/catch_lights/2",
    "/board/catch_tiles_remaining",
    "/board/coins",
    "/board/collision_bounce_behavior",
    "/board/collision_resolution_state",
    "/board/creature_hit_count",
    "/board/event_timer_raw",
    "/board/evolution_arrow_progress",
    "/board/evolution_items_caught",
    "/board/evolution_shop_active",
    "/board/field",
    "/board/flipper_ball_side/0",
    "/board/flipper_ball_side/1",
    "/board/flipper_collision_frame/0",
    "/board/flipper_collision_frame/1",
    "/board/flipper_collision_map_frame/0",
    "/board/flipper_collision_map_frame/1",
    "/board/flipper_direction/0",
    "/board/flipper_direction/1",
    "/board/flipper_position/0",
    "/board/flipper_position/1",
    "/board/hole_lights/0",
    "/board/hole_lights/1",
    "/board/hole_lights/2",
    "/board/hole_lights/3",
    "/board/mode_animation_timer",
    "/board/prize_selected",
    "/board/shop_item_cursor",
    "/board/shop_panel_active",
    "/board/shop_panel_slide",
    "/board/travel_tracker_count",
    "/controls/decision_horizon_seconds",
    "/controls/shadow_fly_action",
    "/game/elapsed_seconds",
    "/game/lives",
    "/game/map",
    "/game/mode",
    "/game/previous_reward",
    "/game/score",
    "/game/stage",
    "/last_transition/reward_terms/ball_loss",
    "/last_transition/reward_terms/energy_cost",
    "/last_transition/reward_terms/progress_reward",
    "/last_transition/reward_terms/score",
    "/last_transition/reward_terms/special_event_bonus",
    "/last_transition/reward_terms/upper_zone",
    "/last_transition/score_delta_points",
    "/last_transition/total_reward_value",
    "/selection_context/confirm_action",
    "/selection_context/cursor",
];

const BOOLEAN_POINTERS: [&str; 32] = [
    "/ball/approaching_flippers",
    "/ball/drain_danger/center_gap_closing",
    "/ball/drain_danger/critical",
    "/ball/drain_danger/game_confirmed_flipper_collision",
    "/ball/drain_danger/moving_down_toward_drain",
    "/ball/drain_danger/rolling_down_flipper_toward_center",
    "/ball/flipper_approach/game_confirmed_flipper_collision",
    "/ball/flipper_approach/left/game_confirmed_contact",
    "/ball/flipper_approach/left/heading_toward",
    "/ball/flipper_approach/left/rolling_toward_inner_tip",
    "/ball/flipper_approach/right/game_confirmed_contact",
    "/ball/flipper_approach/right/heading_toward",
    "/ball/flipper_approach/right/rolling_toward_inner_tip",
    "/ball/flipper_approach/rolling_down_flipper_toward_center",
    "/ball/in_launch_chute",
    "/ball/lost",
    "/ball/physics/free_ball",
    "/ball/physics/game_confirmed_flipper_collision",
    "/ball/physics/supported",
    "/ball/preemptive_projection/will_reach_lower_playfield",
    "/ball/saved_drain",
    "/ball/stationary_on_held_flipper",
    "/board/bonus_field",
    "/board/flipper_active/0",
    "/board/flipper_active/1",
    "/board/flipper_bounce_applied/0",
    "/board/flipper_bounce_applied/1",
    "/board/flipper_held/0",
    "/board/flipper_held/1",
    "/board/flippers_disabled",
    "/game/mode_active",
    "/selection_context/selection_required",
];

#[must_use]
pub fn encode_structured_input(observation: &Value, jev_input: &Value) -> Option<[f32; INPUT_DIM]> {
    let schema = jev_input.get("schema")?.as_str()?;
    if !matches!(
        schema,
        "typesafe-jev-pinball-state-v3" | "typesafe-jev-pinball-state-v4"
    ) {
        return None;
    }

    let mut encoded = [0.0; INPUT_DIM];
    let features = finite_array::<12>(observation.get("features")?)?;
    let objectives = finite_array::<16>(observation.get("objective_features")?)?;
    let proprioception = finite_array::<16>(observation.get("proprio_features")?)?;
    encoded[..12].copy_from_slice(&features);
    encoded[12..28].copy_from_slice(&objectives);
    encoded[28..LEGACY_INPUT_DIM].copy_from_slice(&proprioception);

    for (slot, pointer) in NUMERIC_POINTERS.iter().enumerate() {
        if let Some(value) = jev_input.pointer(pointer) {
            if !value.is_null() {
                encoded[NUMERIC_OFFSET + slot] = finite_f32(value)?;
            }
        }
    }
    encode_event_aggregates(jev_input, &mut encoded)?;

    for (slot, pointer) in BOOLEAN_POINTERS.iter().enumerate() {
        if let Some(value) = jev_input.pointer(pointer) {
            if !value.is_null() {
                encoded[BOOLEAN_OFFSET + slot] = f32::from(value.as_bool()?);
            }
        }
    }

    encode_categories(jev_input, "", &mut encoded);
    encoded
        .iter()
        .all(|value| value.is_finite())
        .then_some(encoded)
}

#[must_use]
pub fn is_important_decision(encoded: &[f32; INPUT_DIM]) -> bool {
    const EVENT_COUNT: usize = NUMERIC_OFFSET + 84;
    const IMPORTANT_BOOLEAN_SLOTS: [usize; 8] = [2, 3, 5, 6, 7, 12, 17, 31];
    encoded[EVENT_COUNT] > 0.0
        || IMPORTANT_BOOLEAN_SLOTS
            .iter()
            .any(|slot| encoded[BOOLEAN_OFFSET + slot] > 0.5)
}

fn encode_event_aggregates(input: &Value, encoded: &mut [f32; INPUT_DIM]) -> Option<()> {
    let Some(events) = input
        .pointer("/last_transition/events")
        .and_then(Value::as_array)
    else {
        return Some(());
    };
    encoded[NUMERIC_OFFSET + 84] = events.len() as f32;
    for event in events {
        encoded[NUMERIC_OFFSET + 85] += optional_f32(event.get("score_points"))?;
        encoded[NUMERIC_OFFSET + 86] += optional_f32(event.get("count"))?;
        encoded[NUMERIC_OFFSET + 87] += optional_f32(event.get("reward_value"))?;
    }
    Some(())
}

fn encode_categories(value: &Value, path: &str, encoded: &mut [f32; INPUT_DIM]) {
    match value {
        Value::Object(values) => {
            for (key, value) in values {
                let next = format!("{path}/{key}");
                encode_categories(value, &next, encoded);
            }
        }
        Value::Array(values) => {
            for value in values {
                let next = format!("{path}/*");
                encode_categories(value, &next, encoded);
            }
        }
        Value::String(category) if !excluded_category(path) => {
            let hash = fnv1a(
                path.as_bytes(),
                fnv1a(category.as_bytes(), 0xcbf2_9ce4_8422_2325),
            );
            let bucket = hash as usize % CATEGORY_SLOTS;
            let sign = if hash & (1_u64 << 63) == 0 { 1.0 } else { -1.0 };
            encoded[CATEGORY_OFFSET + bucket] += sign;
        }
        _ => {}
    }
}

fn excluded_category(path: &str) -> bool {
    path == "/schema"
        || path == "/game/title"
        || path == "/game/control_objective"
        || path.starts_with("/game/coordinate_guide/")
        || path == "/last_transition/reward_schema"
        || path.ends_with("/caveat")
        || path.ends_with("/calibration")
}

fn fnv1a(bytes: &[u8], mut hash: u64) -> u64 {
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

fn optional_f32(value: Option<&Value>) -> Option<f32> {
    value.map_or(Some(0.0), finite_f32)
}

fn finite_array<const N: usize>(value: &Value) -> Option<[f32; N]> {
    let values = value.as_array()?;
    if values.len() != N {
        return None;
    }
    let mut output = [0.0; N];
    for (index, value) in values.iter().enumerate() {
        output[index] = finite_f32(value)?;
    }
    Some(output)
}

fn finite_f32(value: &Value) -> Option<f32> {
    let converted = value.as_f64()? as f32;
    converted.is_finite().then_some(converted)
}
