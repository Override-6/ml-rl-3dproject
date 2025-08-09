use crate::player::Player;
use bevy::color::palettes::css::GREEN;
use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use bevy::prelude::{
    AlignSelf, Commands, Component, PositionType, Query, Res, Text, Transform, Visibility, With,
};
use bevy::text::{JustifyText, TextColor, TextFont, TextLayout};
use bevy::ui::{Node, Val};
use bevy_math::EulerRot;
use bevy_rapier3d::prelude::Velocity;

// New component to mark our UI text
#[derive(Component)]
pub struct StatsText;

#[derive(Component)]
pub struct TriggerZoneText;

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
        TriggerZoneText,
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
