mod map;
mod objective;
mod player;
mod ui;
// unused
mod ai;
mod component;
mod drone;
mod game;
mod human;
mod sensor;

use crate::ai::input_recorder::GameInputRecorder;
use crate::ai::player::{follow_script, spawn_ai_player};
use crate::ai::script::Script;
use crate::component::arrow::{spawn_arrow_resource, spawn_arrows_to_players};
use crate::component::player_character::spawn_player_character;
use crate::game::TICK_RATE;
use crate::human::camera_controller::{camera_follow, mouse_look};
use crate::human::player::{move_player, spawn_human_player};
use crate::map::setup_map;
use crate::objective::{InTriggerZone, check_trigger_zone};
use crate::sensor::ground_sensor::ground_sensor_events;
use crate::sensor::player_vibrissae::{debug_render_lasers, update_vibrissae_lasers};
use crate::ui::{setup_ui, update_stats_text};
use bevy::ecs::schedule::ScheduleLabel;
use bevy::pbr::DirectionalLightShadowMap;
use bevy::prelude::*;
use bevy_rapier3d::prelude::*;
use bevy_rapier3d::rapier::dynamics::IntegrationParameters;
use bincode::encode_into_slice;
use std::cmp::PartialEq;
use std::fs::File;
use std::io::{Read, Write};

#[derive(ScheduleLabel, Clone, Hash, PartialEq, Eq, Debug)]
pub struct PostPhysics;

fn main() {
    let script = read_script_from_file("script.bin");
    let app = create_app(true, script);
    run_simulation(app)
}

fn read_script_from_file(file: &str) -> Option<Script> {
    let file = File::open(file);
    let mut script = None;
    if let Ok(mut file) = file {
        let mut script_data = Vec::new();
        file.read_to_end(&mut script_data).unwrap();
        let decoded_script: Script =
            bincode::decode_from_slice(&script_data, bincode::config::standard())
                .unwrap()
                .0;
        script = Some(decoded_script);
    }

    script
}

fn create_app(head: bool, script: Option<Script>) -> App {
    let mut app = App::new();
    app.add_systems(FixedUpdate, update_vibrissae_lasers);

    if head {
        app.add_plugins(DefaultPlugins.set(AssetPlugin {
            watch_for_changes_override: Some(true),
            ..default()
        }));
        app.add_plugins((RapierDebugRenderPlugin::default(),))
            .add_systems(Update, (mouse_look, camera_follow, update_stats_text))
            .add_systems(
                FixedUpdate,
                debug_render_lasers.after(update_vibrissae_lasers),
            );
    } else {
        app.add_plugins(MinimalPlugins);
    }

    app.add_plugins(RapierPhysicsPlugin::<NoUserData>::default())
        .add_systems(Startup, (setup, setup_ui, setup_map, spawn_arrow_resource))
        .add_systems(
            PostStartup,
            (
                spawn_player_character,
                spawn_arrows_to_players.after(spawn_player_character),
            ),
        )
        .add_systems(FixedUpdate, (ground_sensor_events, check_trigger_zone))
        .add_systems(Update, cleanup_on_exit)
        .insert_resource(
            RapierContextInitialization::InitializeDefaultRapierContext {
                integration_parameters: IntegrationParameters {
                    dt: 1.0 / TICK_RATE,
                    warmstart_coefficient: 0.0,
                    ..default()
                },
                rapier_configuration: RapierConfiguration {
                    ..RapierConfiguration::new(100f32)
                },
            },
        )
        .insert_resource(DirectionalLightShadowMap { size: 100 })
        .insert_resource(Time::<Fixed>::from_hz(TICK_RATE as f64));

    if let Some(script) = script {
        app.insert_resource(script);
        app.add_systems(Startup, spawn_ai_player);
        app.add_systems(FixedUpdate, follow_script);
    } else {
        app.insert_resource(GameInputRecorder::new());
        app.add_systems(Startup, spawn_human_player);
        app.add_systems(FixedUpdate, move_player);
    }
    app
}

fn run_simulation(mut app: App) {}

fn setup(mut commands: Commands) {
    commands.insert_resource(InTriggerZone(false));
}

fn cleanup_on_exit(
    mut events: EventReader<AppExit>,
    recorded_input: Option<Res<GameInputRecorder>>,
) -> Result<(), BevyError> {
    if let Some(event) = events.read().next() {
        info!("App is exiting with status: {:?}", event);

        let Some(recorded_input) = recorded_input else {
            return Ok(());
        };

        let recorded_input: Script = recorded_input.as_ref().into();

        let mut file = File::create("script.bin")?;
        let mut buff = [0u8; 100000];
        encode_into_slice(&recorded_input, &mut buff, bincode::config::standard())?;
        file.write_all(&buff)?;
    }
    Ok(())
}
