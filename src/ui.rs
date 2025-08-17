use crate::component::player_character::PlayerInfoUI;
use crate::player::Player;
use crate::simulation::{PlayerEvaluation, SimulationStepState};
use bevy::color::palettes::basic::RED;
use bevy::color::palettes::css::{BLUE, GREEN};
use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use bevy::prelude::{AlignSelf, Camera, Camera3d, ChildOf, Commands, Component, GlobalTransform, PositionType, Query, Res, Text, Transform, Vec3, Visibility, Window, With};
use bevy::text::{JustifyText, TextColor, TextFont, TextLayout};
use bevy::ui::{Node, Val};
use bevy_math::EulerRot;
use bevy_rapier3d::prelude::Velocity;
use crate::ai::model_control_pipeline::ModelCommands;

// New component to mark our UI text
#[derive(Component)]
pub struct StatsText;

#[derive(Component)]
pub struct SuccessUIText;

pub fn setup_ui(mut commands: Commands) {
    commands.spawn((
        Text::from("FPS: \nPosition: \nRotation: \nVelocity:"),
        TextFont {
            font_size: 20.0,
            ..Default::default()
        },
        StatsText,
    ));

    commands.spawn((
        Text::from("SUCCESS !"),
        TextFont {
            font_size: 30.0,
            ..Default::default()
        },
        TextColor(GREEN.into()),
        TextLayout::new_with_justify(JustifyText::Center),
        Node {
            align_self: AlignSelf::Center,
            position_type: PositionType::Absolute,
            width: Val::Percent(100.0),
            bottom: Val::Px(20.0),
            ..Default::default()
        },
        Visibility::Hidden,
        SuccessUIText,
    ));
}

pub fn update_stats_text(
    player_query: Query<(&Transform, &Velocity), With<Player>>,
    mut text_query: Query<&mut Text, With<StatsText>>,
    diagnostic: Res<DiagnosticsStore>,
) {
    let Some((transform, velocity)) = player_query.iter().next() else {
        return;
    };
    let Ok(mut text) = text_query.single_mut() else {
        return;
    };

    let position = transform.translation;
    let rotation = transform.rotation.to_euler(EulerRot::XYZ);
    let lin_vel = velocity.linvel;
    let ang_vel = velocity.angvel;

    let fps = diagnostic.get(&FrameTimeDiagnosticsPlugin::FPS);

    text.0 = format!(
        "FPS:{}\n\
        Position: {:>5.1}, {:>5.1}, {:>5.1}\n\
        Rotation: {:>5.1}, {:>5.1}, {:>5.1}\n\
        Velocity: {:>5.1}, {:>5.1}, {:>5.1}\n\
        Angular: {:>5.1}, {:>5.1}, {:>5.1}",
        fps.and_then(|fps| fps.average()).unwrap_or(-1.0),
        position.x,
        position.y,
        position.z,
        rotation.0.to_degrees(),
        rotation.1.to_degrees(),
        rotation.2.to_degrees(),
        lin_vel.x,
        lin_vel.y,
        lin_vel.z,
        ang_vel.x,
        ang_vel.y,
        ang_vel.z
    );
}

pub fn update_player_info(
    windows: Query<&Window>,
    camera_q: Query<(&Camera, &GlobalTransform), With<Camera3d>>,
    gt_q: Query<(&Player, &GlobalTransform)>,
    mut player_info_q: Query<
        (&mut Node, &mut Text, &mut TextColor, &PlayerInfoUI),
    >,
    sim_step: Res<SimulationStepState>,
    model_commands: Res<ModelCommands>,
) -> Result<(), bevy::prelude::BevyError> {
    if model_commands.current_step_is_reset {
        return Ok(()); // do not update player info during environment reset
    }
    let window = windows.single()?;
    let (camera, camera_transform) = camera_q.single()?;

    for (mut node, mut text, mut text_color, pui) in
        player_info_q.iter_mut()
    {
        let (player, player_transform) = gt_q.get(pui.0)?;
        // World position above the player
        let world_pos = player_transform.translation() + Vec3::Y * 1.5;

        // Convert to NDC (Normalized Device Coordinates)
        if let Some(ndc) = camera.world_to_ndc(camera_transform, world_pos) {
            if ndc.z > 0.0 {
                // Convert NDC to screen coords
                let screen_x = (ndc.x + 1.0) / 2.0 * window.width();
                let screen_y = (1.0 - ndc.y) / 2.0 * window.height();

                node.left = Val::Px(screen_x);
                node.top = Val::Px(screen_y);
            }
        }

        let PlayerEvaluation { reward, done } = sim_step.player_states[player.id].evaluation;

        if done {
            text_color.0 = BLUE.into();
            text.0 = String::from("DONE");
            continue
        }

        text.0 = format!("{reward}");
        if reward >= 0.0 {
            text_color.0 = GREEN.into();
        } else {
            text_color.0 = RED.into();
        }
    }

    Ok(())
}
