#![feature(const_fn_floating_point_arithmetic)]
#![feature(slice_flatten)]

use core::time;
use std::{f32::consts::PI, ops::Sub, time::{Duration, Instant}};

use image::{ImageBuffer, Rgb};

mod vector;

use rayon::prelude::{IntoParallelRefMutIterator, IndexedParallelIterator, ParallelIterator};
use vector::*;

const WIDTH: u32 = 1024;
const HEIGHT: u32 = 1024;

const MAX_STEPS: u32 = 320;
const MIN_DIST: f32 = 4.0;
const MAX_DIST: f32 = 10.0;
const SURF_DIST: f32 = 0.00025;
const EPSILON: f32 = SURF_DIST * 2.0;

const UP: Vector<3, f32> = vec3(0.0, 1.0, 0.0);
const DOWN: Vector<3, f32> = vec3(0.0, -1.0, 0.0);

const fn deg2rad(theta: f32) -> f32 {
    theta * PI / 180.0
}

struct Gem {
    table: f32,
    culet: f32,
    girdle_facets: u8,
    radius: f32,
}

fn sdf_sphere(p: Vector<3, f32>, radius: f32) -> f32 {
    length(p) - radius
}

fn sdf_cylinder(p: Vector<3, f32>, radius: f32) -> f32 {
    length(vec2(p.x(), p.z())) - radius
}

fn sdf_plane(p: Vector<3, f32>, normal: Vector<3, f32>, h: f32) -> f32 {
    dot(p, normalize(normal)) - h
}

fn sdf_nprism(p: Vector<3, f32>, num_facets: u8, radius: f32) -> f32 {
    let theta_n = PI * 2.0 / num_facets as f32;
    let mut theta = p.z().atan2(p.x());
    if theta < 0.0 {
        theta += PI * 2.0;
    }
    theta = ((theta + 0.5 * theta_n).rem_euclid(theta_n) - theta_n * 0.5).abs();

    let mag = length(vec2(p.x(), p.z()));
    let p2 = mag * vec2(theta.cos(), theta.sin());
    let y = radius * f32::tan(0.5 * theta_n);
    if p2.y() > y {
        length(vec2(radius, y) - p2)
    } else {
        p2.x() - radius
    }
}

fn sdf_ncone(p: Vector<3, f32>, n: u8, radius: f32, azimuth: f32, elevation: f32) -> f32 {
    let elevation = deg2rad(90.0 - elevation);
    let azimuth = deg2rad(azimuth);
    let theta = p.z().atan2(p.x());
    let magnitude = length(vec2(p.x(), p.z()));
    
    let theta = (theta - PI / n as f32 - azimuth).rem_euclid(2.0 * PI / n as f32) - PI / n as f32;
    let p2 = magnitude * vec2(theta.sin(), theta.cos());

    let d = p2.y() - p.y() * elevation.tan() - radius / elevation.cos();
    d * elevation.cos()
}

fn sdf_gemstone(p: Vector<3, f32>, gem: Gem) -> f32 {
    let mut dist = if gem.girdle_facets < 3 {
        sdf_cylinder(p, gem.radius)
    } else {
        sdf_nprism(p, gem.girdle_facets, gem.radius)
    };

    dist = dist.max(sdf_plane(p, UP, gem.table));
    // dist = dist.max(sdf_plane(p, DOWN, gem.culet));

    // crown
    dist = dist.max(sdf_ncone(p, 16, 4.6, 0.0/96.0*360.0, 50.0));
    dist = dist.max(sdf_ncone(p, 16, 4.175, 3.0/96.0*360.0, 43.0));
    dist = dist.max(sdf_ncone(p, 16, 4.13, 6.0/96.0*360.0, 41.0));

    // pavilion
    dist = dist.max(sdf_ncone(p, 16, 4.7, 0.0/96.0*360.0, 180.0 - 50.0));
    dist = dist.max(sdf_ncone(p, 16, 4.365, 3.0/96.0*360.0, 180.0 - 44.0));
    dist = dist.max(sdf_ncone(p, 16, 3.94, 6.0/96.0*360.0, 180.0 - 35.0));
    dist = dist.max(sdf_ncone(p, 16, 3.37, 0.0/96.0*360.0, 180.0 - 24.0));

    // dist = dist.max(sdf_ncone(p, 8, 3.0, 0.0, 38.0));
    // dist = dist.max(sdf_ncone(p, 8, 1.8375, 6.0/96.0*360.0, 24.0));
    // dist = dist.max(sdf_ncone(p, 4, 1.35, 0.0, 16.0));
    // dist = dist.max(sdf_ncone(p, 8, 3.0, 0.0, 180.0 - 38.0));
    dist = dist.max(sdf_sphere(p, 7.0));

    dist
}

fn get_dist(p: Vector<3, f32>) -> f32 {
    let mut dist = f32::MAX;
    // dist = f32::min(dist, sdf_sphere(p, 5.0));
    dist = dist.min(sdf_gemstone(p, Gem {
        culet: 5.8,
        girdle_facets: 16,
        radius: 6.0,
        table: 1.75,
    }));
    dist
}

fn get_normal(p: Vector<3, f32>) -> Vector<3, f32> {
    let d = get_dist(p);
    normalize(d - vec3(
        get_dist(p - vec3(EPSILON, 0.0, 0.0)),
        get_dist(p - vec3(0.0, EPSILON, 0.0)),
        get_dist(p - vec3(0.0, 0.0, EPSILON))
    ))
}

fn ray_sphere_intersection(ray_origin: Vector<3, f32>, ray_dir: Vector<3, f32>, radius: f32) -> Option<Vector<3, f32>> {
    let a = dot(ray_dir, ray_dir);
    let b = 2.0 * dot(ray_origin, ray_dir);
    let c = dot(ray_origin, ray_origin) - radius * radius;

    let discriminant = b * b - 4.0 * a * c;
    if discriminant < 0.0 {
        None
    } else {
        let t = (-b + discriminant.sqrt()) / (2.0 * a);
        Some(ray_origin + ray_dir * t)
    }
}

fn ray_march(ray_origin: Vector<3, f32>, ray_dir: Vector<3, f32>) -> Vector<3, f32> {
    let mut p = ray_origin + MIN_DIST * ray_dir;
    let mut num_bounces = 0;


    let mut i = 0;
    while i < MAX_STEPS {
        let dist = get_dist(p);
        p += ray_dir * dist;

        // if i == 1 {
        //     return vec3(dist, dist, dist);
        // }
        
        // trace initial hit
        if dist < SURF_DIST {
            // break;
            num_bounces += 1;
            let normal = get_normal(p);
            // return normal * - 1.0;


            let fresnel = f32::powf(1.0 + dot(ray_dir, normal), 5.0);

            let refl = reflect(ray_dir, normal);

            if let Some(vec) = ray_sphere_intersection(p, refl, 20.0) {
                return rem(vec / 4.0, vec3(1.0, 1.0, 1.0)) * fresnel;
            } else {
                return vec3(0.0, 0.0, 1.0);
            }

            // return ray_dir;
            return vec3(fresnel, fresnel, fresnel);
        }



        if dist > MAX_DIST {
            if let Some(vec) = ray_sphere_intersection(p, ray_dir, 20.0) {
                return rem(vec / 4.0, vec3(1.0, 1.0, 1.0));
            } else {
                return vec3(0.0, 0.0, 1.0);
            }
        }

        i += 1;
    }

    vec3(1.0, 1.0, 1.0) * i as f32 / MAX_STEPS as f32
}

fn main() {

    let resolution = vec2(WIDTH as f32, HEIGHT as f32);
    let up = vec3(0.0, 1.0, 0.0);



    // println!("{:?}", ray_origin);
    // println!("{:?}", look_at);
    // println!("{:?}", look_dir);
    // println!("{:?}", cam_right);
    // println!("{:?}", cam_up);
    // println!("{:?} x {:?} = {:?} ", cam_right, look_dir, cross(cam_right, look_dir));
    // println!("{:?} dot {:?} = {:?} ", cam_right, cam_right, dot(cam_right, -1.0 * cam_right));

    for theta in 0..1 {
        
        let f = theta as f32;
        let f = 2.0 * PI * f / 360.0;
        
        let camera_dist = 10.0f32;
        let ray_origin = vec3(-camera_dist * f.cos(), 0.0, camera_dist * f.sin());
        let look_at = vec3(0.0, 0.0, 0.0);
        let look_dir = normalize(look_at - ray_origin);
        let cam_right = cross(look_dir, up);
        let cam_up = cross(cam_right, look_dir);

        let t0 = Instant::now();
        let mut buffer = Box::new([[[0u8, 0, 0]; WIDTH as usize]; HEIGHT as usize]);
        buffer.par_iter_mut()
            .enumerate()
            .for_each(|(y, row)| 
                row.iter_mut()
                    .enumerate()
                    .for_each(|(x, pixel)| {
                        let frag_coord = vec2(x as f32, HEIGHT.sub(y as u32).sub(1) as f32);
                        let uv = 2.0 * (frag_coord - 0.5 * resolution) / resolution.y();
                        let ds = 2.0 / resolution;
    
                        let mut col = vec3(0.0, 0.0, 0.0);
    
                        let ray_dir = normalize(uv.x() * cam_right + uv.y() * cam_up + look_dir);
    
                        let color = ray_march(ray_origin, ray_dir);
                        *pixel = color.to_rgb_u8();
                    })
            );
        
        println!("Time: {:?}", Instant::now() - t0);
    
        image::save_buffer(format!("out{:03}.png", theta), buffer.flatten().flatten(), WIDTH as u32, HEIGHT as u32, image::ColorType::Rgb8).expect("failed to save img");
    }

    // image::open("out.png").unwrap().resize(WIDTH / 2, HEIGHT / 2, image::imageops::FilterType::Gaussian).save("out_scaled.png").unwrap();
    // image::open("out_scaled.png").unwrap().resize(WIDTH / 4, HEIGHT / 4, image::imageops::FilterType::Gaussian).save("out_scaled2.png").unwrap();

    println!("Hello, world!");
}
