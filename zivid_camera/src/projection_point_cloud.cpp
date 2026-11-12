// Copyright 2026 Zivid AS
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//    * Redistributions of source code must retain the above copyright
//      notice, this list of conditions and the following disclaimer.
//
//    * Redistributions in binary form must reproduce the above copyright
//      notice, this list of conditions and the following disclaimer in the
//      documentation and/or other materials provided with the distribution.
//
//    * Neither the name of the Zivid AS nor the names of its
//      contributors may be used to endorse or promote products derived from
//      this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <zivid_camera/projection_point_cloud.hpp>
#include <zivid_camera/utility.hpp>

namespace
{
using sensor_msgs::msg::PointField;

const PointField * findField(const sensor_msgs::msg::PointCloud2 & cloud, const std::string & name)
{
  const auto it = std::find_if(cloud.fields.begin(), cloud.fields.end(), [&](const auto & field) {
    return field.name == name;
  });
  return it == cloud.fields.end() ? nullptr : &*it;
}

void validateField(const PointField * field, uint32_t point_step, bool coordinate)
{
  if (!field) {
    throw std::runtime_error("Point cloud must contain x, y and z fields");
  }
  const bool valid_type =
    coordinate ? (field->datatype == PointField::FLOAT32 || field->datatype == PointField::FLOAT64)
               : (field->datatype == PointField::FLOAT32 || field->datatype == PointField::UINT32);
  const size_t size = field->datatype == PointField::FLOAT64 ? 8 : 4;
  if (
    !valid_type || field->count != 1 || field->offset > point_step ||
    size > point_step - field->offset) {
    throw std::runtime_error("Invalid point cloud field: " + field->name);
  }
}

template <typename T>
T readValue(const uint8_t * data, bool big_endian)
{
  std::array<uint8_t, sizeof(T)> bytes;
  std::copy_n(data, bytes.size(), bytes.begin());
  const uint16_t one = 1;
  const bool host_big_endian = *reinterpret_cast<const uint8_t *>(&one) == 0;
  if (big_endian != host_big_endian) {
    std::reverse(bytes.begin(), bytes.end());
  }
  T value;
  std::memcpy(&value, bytes.data(), sizeof(value));
  return value;
}

double readCoordinate(const uint8_t * point, const PointField & field, bool big_endian)
{
  const auto * data = point + field.offset;
  return field.datatype == PointField::FLOAT32 ? readValue<float>(data, big_endian)
                                               : readValue<double>(data, big_endian);
}

void validateIntensityField(const PointField & field, uint32_t point_step)
{
  size_t size = 0;
  switch (field.datatype) {
    case PointField::INT8:
    case PointField::UINT8:
      size = 1;
      break;
    case PointField::INT16:
    case PointField::UINT16:
      size = 2;
      break;
    case PointField::INT32:
    case PointField::UINT32:
    case PointField::FLOAT32:
      size = 4;
      break;
    case PointField::FLOAT64:
      size = 8;
      break;
    default:
      break;
  }
  if (
    size == 0 || field.count != 1 || field.offset > point_step ||
    size > point_step - field.offset) {
    throw std::runtime_error("Invalid point cloud field: " + field.name);
  }
}

double readIntensity(const uint8_t * point, const PointField & field, bool big_endian)
{
  const auto * data = point + field.offset;
  switch (field.datatype) {
    case PointField::INT8:
      return readValue<int8_t>(data, big_endian);
    case PointField::UINT8:
      return readValue<uint8_t>(data, big_endian);
    case PointField::INT16:
      return readValue<int16_t>(data, big_endian);
    case PointField::UINT16:
      return readValue<uint16_t>(data, big_endian);
    case PointField::INT32:
      return readValue<int32_t>(data, big_endian);
    case PointField::UINT32:
      return readValue<uint32_t>(data, big_endian);
    case PointField::FLOAT32:
      return readValue<float>(data, big_endian);
    case PointField::FLOAT64:
      return readValue<double>(data, big_endian);
    default:
      throw std::runtime_error("Invalid point cloud field: " + field.name);
  }
}

Zivid::ColorBGRA intensityColor(double intensity, double min_intensity, double max_intensity)
{
  if (!std::isfinite(intensity)) {
    return Zivid::ColorBGRA{0, 0, 0, 255};
  }
  const auto fraction = min_intensity == max_intensity
                          ? 0.5L
                          : (static_cast<long double>(intensity) - min_intensity) /
                              (static_cast<long double>(max_intensity) - min_intensity);
  const auto channel = [](long double value) {
    return static_cast<uint8_t>(std::lround(255 * std::clamp(value, 0.0L, 1.0L)));
  };
  const auto position = 4 * fraction;
  return Zivid::ColorBGRA{
    channel(2 - position), channel(std::min(position, 4 - position)), channel(position - 2), 255};
}
}  // namespace

namespace zivid_camera
{
tf2::Transform cameraFromCloud(
  const tf2_ros::Buffer & buffer, const std::string & camera_frame,
  const std_msgs::msg::Header & cloud_header)
{
  if (cloud_header.frame_id.empty()) {
    throw std::runtime_error("Point cloud frame_id must not be empty");
  }
  if (cloud_header.frame_id == camera_frame) {
    return tf2::Transform::getIdentity();
  }
  const auto transform = buffer.lookupTransform(
    camera_frame, cloud_header.frame_id, rclcpp::Time{cloud_header.stamp},
    rclcpp::Duration::from_seconds(1.0));
  tf2::Transform result;
  tf2::fromMsg(transform.transform, result);
  return result;
}

ProjectionPoints prepareProjectionPoints(
  const sensor_msgs::msg::PointCloud2 & cloud, const tf2::Transform & camera_from_cloud)
{
  const auto * x = findField(cloud, "x");
  const auto * y = findField(cloud, "y");
  const auto * z = findField(cloud, "z");
  validateField(x, cloud.point_step, true);
  validateField(y, cloud.point_step, true);
  validateField(z, cloud.point_step, true);
  const auto * color = findField(cloud, "rgba");
  if (!color) {
    color = findField(cloud, "rgb");
  }
  if (color) {
    validateField(color, cloud.point_step, false);
  }
  const auto * intensity = color ? nullptr : findField(cloud, "intensity");
  if (intensity) {
    validateIntensityField(*intensity, cloud.point_step);
  }
  const auto row_size = static_cast<uint64_t>(cloud.width) * cloud.point_step;
  const auto data_size = static_cast<uint64_t>(cloud.height) * cloud.row_step;
  if (
    cloud.width == 0 || cloud.height == 0 || row_size > cloud.row_step ||
    data_size > cloud.data.size()) {
    throw std::runtime_error("Invalid or empty point cloud data");
  }

  ProjectionPoints result;
  std::vector<double> intensities;
  for (size_t row = 0; row < cloud.height; ++row) {
    for (size_t col = 0; col < cloud.width; ++col) {
      const auto * point = cloud.data.data() + row * cloud.row_step + col * cloud.point_step;
      const tf2::Vector3 source{
        readCoordinate(point, *x, cloud.is_bigendian),
        readCoordinate(point, *y, cloud.is_bigendian),
        readCoordinate(point, *z, cloud.is_bigendian)};
      if (!std::isfinite(source.x()) || !std::isfinite(source.y()) || !std::isfinite(source.z())) {
        continue;
      }
      const auto transformed = camera_from_cloud * source;
      const Zivid::PointXYZ camera_point{
        rosLengthToZivid(transformed.x()), rosLengthToZivid(transformed.y()),
        rosLengthToZivid(transformed.z())};
      if (
        !std::isfinite(camera_point.x) || !std::isfinite(camera_point.y) ||
        !std::isfinite(camera_point.z) || camera_point.z <= 0.0F) {
        continue;
      }
      result.points.push_back(camera_point);
      if (intensity) {
        intensities.push_back(readIntensity(point, *intensity, cloud.is_bigendian));
      } else {
        const uint32_t rgb =
          color ? readValue<uint32_t>(point + color->offset, cloud.is_bigendian) : 0x00ffffff;
        result.colors.emplace_back(
          static_cast<uint8_t>(rgb & 0xff), static_cast<uint8_t>((rgb >> 8) & 0xff),
          static_cast<uint8_t>((rgb >> 16) & 0xff), 255);
      }
    }
  }
  if (result.points.empty()) {
    throw std::runtime_error("Point cloud contains no finite points in front of the camera");
  }
  if (intensity) {
    double min_intensity = std::numeric_limits<double>::infinity();
    double max_intensity = -std::numeric_limits<double>::infinity();
    for (const auto value : intensities) {
      if (std::isfinite(value)) {
        min_intensity = std::min(min_intensity, value);
        max_intensity = std::max(max_intensity, value);
      }
    }
    for (const auto value : intensities) {
      result.colors.push_back(intensityColor(value, min_intensity, max_intensity));
    }
  }
  return result;
}

Zivid::Image<Zivid::ColorBGRA> renderProjectionPoints(
  const Zivid::Resolution & resolution, const std::vector<Zivid::PointXY> & pixels,
  const std::vector<Zivid::ColorBGRA> & colors)
{
  if (pixels.size() != colors.size()) {
    throw std::runtime_error("Projector pixel and color counts do not match");
  }
  std::vector<Zivid::ColorBGRA> image(resolution.size(), Zivid::ColorBGRA{0, 0, 0, 255});
  for (size_t i = 0; i < pixels.size(); ++i) {
    const auto & pixel = pixels[i];
    if (
      !std::isfinite(pixel.x) || !std::isfinite(pixel.y) || pixel.x < 0 || pixel.y < 0 ||
      static_cast<double>(pixel.x) >= static_cast<double>(resolution.width()) ||
      static_cast<double>(pixel.y) >= static_cast<double>(resolution.height())) {
      continue;
    }
    const auto x = static_cast<size_t>(pixel.x);
    const auto y = static_cast<size_t>(pixel.y);
    image[y * resolution.width() + x] = colors[i];
  }
  return Zivid::Image<Zivid::ColorBGRA>{resolution, image.begin(), image.end()};
}
}  // namespace zivid_camera
