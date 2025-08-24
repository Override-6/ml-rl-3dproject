use crate::ai::evaluation::PlayerEvaluation;
use crate::component::player_character::PlayerInfoUI;
use crate::player::Player;
use crate::simulation::SimulationState;
use bevy::color::palettes::basic::RED;
use bevy::color::palettes::css::{GRAY, GREEN};
use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use bevy::prelude::{
    AlignItems, Camera, Camera3d, Commands, Component, Display, GlobalTransform, JustifyContent,
    PositionType, Query, Res, Text, Vec3, Window, With, Without,
};
use bevy::text::{TextColor, TextFont};
use bevy::ui::{Node, Val};

#[derive(Component)]
pub struct StatsText;

#[derive(Component)]
pub struct SimulationEpochText;

pub fn setup_ui(mut commands: Commands) {
    commands.spawn((
        Text::from("FPS: {}"),
        TextFont {
            font_size: 20.0,
            ..Default::default()
        },
        StatsText,
    ));
    commands
        .spawn((Node {
            position_type: PositionType::Absolute,
            width: Val::Percent(100.0),
            height: Val::Percent(100.0),
            justify_content: JustifyContent::Center,
            align_items: AlignItems::FlexStart,
            ..Default::default()
        },))
        .with_children(|parent| {
            parent.spawn((
                Text::from("Simulation {}"),
                TextFont {
                    font_size: 40.0,
                    ..Default::default()
                },
                SimulationEpochText,
                Node {
                    position_type: PositionType::Absolute,
                    top: Val::Px(20.0),
                    ..Default::default()
                },
            ));
        });
}

pub fn update_stats_text(
    mut transform_ui_query: Query<&mut Text, (With<StatsText>, Without<SimulationEpochText>)>,
    mut epoch_ui_query: Query<&mut Text, (With<SimulationEpochText>, Without<StatsText>)>,
    diagnostic: Res<DiagnosticsStore>,
    sim: Res<SimulationState>,
) {
    let Ok(mut stats_text) = transform_ui_query.single_mut() else {
        return;
    };

    let Ok(mut epoch_text) = epoch_ui_query.single_mut() else {
        return;
    };

    epoch_text.0 = format!("Simulation {}", sim.epoch + 1);

    let fps = diagnostic.get(&FrameTimeDiagnosticsPlugin::FPS);

    stats_text.0 = format!("FPS:{}", fps.and_then(|fps| fps.average()).unwrap_or(-1.0),);
}

pub fn update_player_info(
    windows: Query<&Window>,
    camera_q: Query<(&Camera, &GlobalTransform), With<Camera3d>>,
    gt_q: Query<(&Player, &GlobalTransform)>,
    mut player_info_q: Query<(&mut Node, &mut Text, &mut TextColor, &PlayerInfoUI)>,
    sim: Res<SimulationState>,
) -> Result<(), bevy::prelude::BevyError> {
    if sim.resetting || sim.timestep == 0 {
        return Ok(()); // do not update player info during environment reset
    }
    let window = windows.single()?;
    let (camera, camera_transform) = camera_q.single()?;

    for (mut node, mut text, mut text_color, pui) in player_info_q.iter_mut() {
        let (player, player_transform) = gt_q.get(pui.0)?;

        if !sim.debug.print_players_rewards {
            node.display = Display::None;
            continue;
        } else {
            node.display = Display::default();
        }

        // World position above the player
        let world_pos = player_transform.translation() + Vec3::Y * 1.5;

        // Convert to NDC (Normalized Device Coordinates)
        if let Some(ndc) = camera.world_to_ndc(camera_transform, world_pos)
            && ndc.z > 0.0
        {
            // Convert NDC to screen coords
            let screen_x = (ndc.x + 1.0) / 2.0 * window.width();
            let screen_y = (1.0 - ndc.y) / 2.0 * window.height();

            node.left = Val::Px(screen_x);
            node.top = Val::Px(screen_y);
        }

        let PlayerEvaluation { reward, done } =
            sim.current_step_state().player_states[player.id].evaluation;

        text.0 = format!("{reward:.2}");

        text_color.0 = if done {
            GRAY.into()
        } else if reward >= 0.0 {
            GREEN.into()
        } else {
            RED.into()
        }
    }

    Ok(())
}
