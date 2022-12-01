from datetime import datetime, timedelta
from typing import MutableSequence
from xml.etree import ElementTree

import cv2 as cv


class XmlPars:
    __tree: ElementTree
    __start_children: int = 1
    __end_children: int = 3
    __locations: ElementTree.Element | MutableSequence[ElementTree.Element] = []

    def __init__(self, xml_path: str):
        self.__tree = ElementTree.parse(xml_path).getroot()

    @staticmethod
    def change_str_to_datetime(time: str) -> datetime:
        return datetime.strptime(time, '%Y-%m-%dT%H:%M:%S.%fZ')

    def get_start_datetime(self) -> datetime:
        return self.change_str_to_datetime(time=self.__tree[1][0].text)

    def __get_closest_index(self, time: datetime) -> int:

        if abs(self.change_str_to_datetime(self.__locations[0][0].text) - time) < abs(
                self.change_str_to_datetime(self.__locations[1][0].text) - time):
            return 0
        else:
            return 1

    def get_current_location(self, time: datetime) -> str | None:
        self.__locations = self.__tree[self.__start_children:self.__end_children]

        if len(self.__locations) == 0:
            return
        elif len(self.__locations) == 1:
            return self.__locations[0].get("lat")
        else:
            index = self.__get_closest_index(time=time)

            if index == 0:
                return self.__locations[0].get("lat")
            else:
                self.__start_children += 1
                self.__end_children += 1
                return self.__locations[1].get("lat")


def get_time(milliseconds):
    seconds_t = milliseconds // 1000
    minutes_t = 0
    hours_t = 0

    if seconds_t >= 60:
        minutes_t = seconds_t // 60
        seconds_t = seconds_t % 60

    if minutes_t >= 60:
        hours_t = minutes_t // 60
        minutes_t = minutes_t % 60

    return hours_t, minutes_t, seconds_t


def run_detection():
    xml = XmlPars(xml_path='2022-11-30T10_11_08.310580.xml')
    start_datetime = xml.get_start_datetime()

    capture = cv.VideoCapture('2022-11-30T10_11_08.310580.mp4')

    while True:
        grabbed, frame = capture.read()
        if not grabbed:
            break

        milliseconds = capture.get(cv.CAP_PROP_POS_MSEC)

        if milliseconds > 0:
            hours, minutes, seconds = get_time(milliseconds=milliseconds)
            current_time = start_datetime + timedelta(hours=hours, minutes=minutes, seconds=seconds)
            print(xml.get_current_location(current_time))

        cv.imshow('frame', frame)

        if cv.waitKey(1) == ord('q'):
            break

    capture.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    run_detection()
