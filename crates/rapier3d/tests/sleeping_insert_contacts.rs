//! Regression test: bodies inserted with `RigidBodyBuilder::sleeping(true)` must not sink through
//! their resting contacts after waking (stale `PAIR_HINT_DYN_BIT` in the contact-recycle path).

use rapier3d::prelude::*;

fn max_sink(n: usize, asleep: bool, ticks: usize) -> f32 {
    let mut w = PhysicsWorld::default();
    w.gravity = Vector::new(0.0, -9.81, 0.0);
    w.insert(RigidBodyBuilder::fixed().translation(Vector::new(0.0, -0.5, 0.0)),
             ColliderBuilder::cuboid(20.0, 0.5, 20.0).friction(0.5));
    let h = 0.35f32;
    let mut bs = vec![];
    for k in 0..n {
        let rb = RigidBodyBuilder::dynamic()
            .translation(Vector::new(0.0, h + 2.0 * h * k as f32, 0.0))
            .sleeping(asleep);
        bs.push(w.insert(rb, ColliderBuilder::cuboid(h, h, h).density(200.0).friction(0.5)).0);
    }
    let mut worst = 0.0f32;
    for _ in 0..ticks {
        w.step();
        for (k, &b) in bs.iter().enumerate() {
            worst = worst.max((h + 2.0 * h * k as f32) - w.bodies[b].translation().y);
        }
    }
    worst
}

#[test]
fn sleeping_spawn_matches_awake_spawn() {
    for n in [1usize, 2, 10] {
        let (awake, asleep) = (max_sink(n, false, 120), max_sink(n, true, 120));
        // one step of free fall (g dt^2 = 2.7 mm at 60 Hz) is tolerated
        assert!(asleep <= awake + 0.0028, "n={n}: awake {awake} asleep {asleep}");
    }
}
