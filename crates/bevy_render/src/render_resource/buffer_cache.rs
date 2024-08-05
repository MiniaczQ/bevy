use crate::renderer::RenderDevice;
use bevy_ecs::{prelude::ResMut, system::Resource};
use bevy_utils::{Entry, HashMap};
use wgpu::{util::BufferInitDescriptor, BindingResource, BufferDescriptor};

use super::{Buffer, IntoBinding};

struct CachedBufferMeta {
    buffer: Buffer,
    taken: bool,
    frames_since_last_use: usize,
}

#[derive(Clone)]
pub struct CachedBuffer {
    pub buffer: Buffer,
}

impl<'a> IntoBinding<'a> for &'a CachedBuffer {
    #[inline]
    fn into_binding(self) -> BindingResource<'a> {
        BindingResource::Buffer(self.buffer.as_entire_buffer_binding())
    }
}

#[derive(Resource, Default)]
pub struct BufferCache {
    buffers: HashMap<BufferDescriptor<'static>, Vec<CachedBufferMeta>>,
}

impl BufferCache {
    pub fn get(
        &mut self,
        render_device: &RenderDevice,
        descriptor: BufferDescriptor<'static>,
    ) -> CachedBuffer {
        match self.buffers.entry(descriptor) {
            Entry::Occupied(mut entry) => {
                for buffer in entry.get_mut().iter_mut() {
                    if !buffer.taken {
                        buffer.frames_since_last_use = 0;
                        buffer.taken = true;
                        return CachedBuffer {
                            buffer: buffer.buffer.clone(),
                        };
                    }
                }

                let buffer = render_device.create_buffer(&entry.key().clone());
                entry.get_mut().push(CachedBufferMeta {
                    buffer: buffer.clone(),
                    frames_since_last_use: 0,
                    taken: true,
                });
                CachedBuffer { buffer }
            }
            Entry::Vacant(entry) => {
                let buffer = render_device.create_buffer(entry.key());
                entry.insert(vec![CachedBufferMeta {
                    buffer: buffer.clone(),
                    taken: true,
                    frames_since_last_use: 0,
                }]);
                CachedBuffer { buffer }
            }
        }
    }

    pub fn get_or(
        &mut self,
        render_device: &RenderDevice,
        descriptor: BufferDescriptor<'static>,
        callback: impl FnOnce() -> Vec<u8>,
    ) -> CachedBuffer {
        match self.buffers.entry(descriptor) {
            Entry::Occupied(mut entry) => {
                for buffer in entry.get_mut().iter_mut() {
                    if !buffer.taken {
                        buffer.frames_since_last_use = 0;
                        buffer.taken = true;
                        return CachedBuffer {
                            buffer: buffer.buffer.clone(),
                        };
                    }
                }

                let descriptor = entry.key().clone();
                let descriptor = BufferInitDescriptor {
                    label: descriptor.label,
                    contents: &callback(),
                    usage: descriptor.usage,
                };

                let buffer = render_device.create_buffer_with_data(&descriptor);
                entry.get_mut().push(CachedBufferMeta {
                    buffer: buffer.clone(),
                    frames_since_last_use: 0,
                    taken: true,
                });
                CachedBuffer { buffer }
            }
            Entry::Vacant(entry) => {
                let descriptor = entry.key().clone();
                let descriptor = BufferInitDescriptor {
                    label: descriptor.label,
                    contents: &callback(),
                    usage: descriptor.usage,
                };

                let buffer = render_device.create_buffer_with_data(&descriptor);
                entry.insert(vec![CachedBufferMeta {
                    buffer: buffer.clone(),
                    frames_since_last_use: 0,
                    taken: true,
                }]);
                CachedBuffer { buffer }
            }
        }
    }

    pub fn update(&mut self) {
        for buffers in self.buffers.values_mut() {
            for buffer in buffers.iter_mut() {
                buffer.frames_since_last_use += 1;
                buffer.taken = false;
            }

            buffers.retain(|texture| texture.frames_since_last_use < 3);
        }
    }
}

pub fn update_buffer_cache_system(mut buffer_cache: ResMut<BufferCache>) {
    buffer_cache.update();
}
