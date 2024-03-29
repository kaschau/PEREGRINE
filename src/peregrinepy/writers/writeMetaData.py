from lxml import etree
from copy import deepcopy


class gridMetaData:
    def __init__(self, precision, lump):
        self.metaType = "grid"
        self.lump = lump
        self.precision = precision

        # This is the main xdmf object
        self.tree = etree.Element("Xdmf")
        self.tree.set("Version", "2")

        self.domainElem = etree.SubElement(self.tree, "Domain")
        self.gridElem = etree.SubElement(self.domainElem, "Grid")
        self.gridElem.set("Name", "PEREGRINE Output")
        self.gridElem.set("GridType", "Collection")
        self.gridElem.set("CollectionType", "Spatial")

        # This is a template of an individual block
        self.blockTemplate = etree.Element("Grid")
        self.blockTemplate.set("Name", "B#Here")

        topologyElem = etree.SubElement(self.blockTemplate, "Topology")
        topologyElem.set("TopologyType", "3DSMesh")
        topologyElem.set("NumberOfElements", "Num Elem Here")
        geometryElem = etree.SubElement(self.blockTemplate, "Geometry")
        geometryElem.set("GeometryType", "X_Y_Z")

        dataXElem = etree.SubElement(geometryElem, "DataItem")
        dataXElem.set("NumberType", "Float")
        dataXElem.set("Dimensions", "XYZ Dims Here")
        dataXElem.set("Precision", "8" if precision == "double" else "4")
        dataXElem.set("Format", "HDF")

        dataXElem.text = "gridFile location:/coordinate/x"
        geometryElem.append(deepcopy(dataXElem))
        geometryElem[-1].text = "gridFile location:/coordinates/y"

        geometryElem.append(deepcopy(dataXElem))
        geometryElem[-1].text = "gridFile location:/coordinates/z"

    def saveXdmf(self, path="./", nrt=None):
        et = etree.ElementTree(self.tree)
        saveFile = f"{path}/{self.getOutputName(nrt=nrt)}"
        et.write(saveFile, pretty_print=True, encoding="UTF-8", xml_declaration=True)

    def addBlockElem(self, nblki, ni, nj, nk, ng):
        blockElem = deepcopy(self.blockTemplate)
        blockElem.set("Name", f"B{nblki:06d}")
        topo = blockElem.find("Topology")
        topo.set("NumberOfElements", f"{nk+2*ng} {nj+2*ng} {ni+2*ng}")

        for coord, i in zip(["x", "y", "z"], [0, 1, 2]):
            X = blockElem.find("Geometry")[i]
            X.set("Dimensions", f"{nk+2*ng} {nj+2*ng} {ni+2*ng}")
            X.text = self.getGridFileH5Location(coord, nblki)

        self.gridElem.append(deepcopy(blockElem))

    def getGridFileH5Location(self, coord, nblki):
        if self.metaType == "grid":
            gridPath = "."
        else:
            gridPath = self.gridPath

        if self.lump:
            return f"{gridPath}/g.h5:/coordinates_{nblki:06d}/{coord}"
        else:
            return f"{gridPath}/g.{nblki:06d}.h5:/coordinates_{nblki:06d}/{coord}"

    def getGridFileName(self, coord, nblki):
        if self.metaType == "grid":
            gridPath = "."
        else:
            gridPath = self.gridPath

        if self.lump:
            return f"{gridPath}/g.h5"
        else:
            return f"{gridPath}/g.{nblki:06d}.h5"

    def getOutputName(self, **kwags):
        return "g.xmf"


class restartMetaData(gridMetaData):
    def __init__(self, gridPath, precision, animate, lump, nrt=0, tme=0.0):
        super().__init__(precision, lump)
        self.metaType = "restart"

        self.animate = animate
        self.gridPath = gridPath

        self.timeElem = etree.SubElement(self.blockTemplate, "Time")
        self.timeElem.set("Value", str(tme))

        # Scalar attribute template
        self.scalarAttributeTemplate = etree.Element("Attribute")
        self.scalarAttributeTemplate.set("Name", "var name here")
        self.scalarAttributeTemplate.set("ScalarAttributeType", "Scalar")
        self.scalarAttributeTemplate.set("Center", "Cell")

        # Vector attribute template
        self.vectorAttributeTemplate = etree.Element("Attribute")
        self.vectorAttributeTemplate.set("Name", "vector name")
        self.vectorAttributeTemplate.set("AttributeType", "Vector")
        self.vectorAttributeTemplate.set("Center", "Cell")
        function = etree.SubElement(self.vectorAttributeTemplate, "DataItem")
        function.set("ItemType", "Function")
        function.set("Function", "JOIN($0, $1, $2)")
        function.set("Dimensions", "Dimension here 3")

        # Data Item template
        self.dataItemTemplate = etree.Element("DataItem")
        self.dataItemTemplate.set("NumberType", "Float")
        self.dataItemTemplate.set("Dimensions", "var dim nums here")
        self.dataItemTemplate.set("Precision", "8" if precision == "double" else "4")
        self.dataItemTemplate.set("Format", "HDF")
        self.dataItemTemplate.text = "resultFile location:/results/"

    def addBlockElem(self, nblki, ni, nj, nk, ng):
        blockElem = deepcopy(self.blockTemplate)
        blockElem.set("Name", f"B{nblki:06d}")
        topo = blockElem.find("Topology")
        topo.set("NumberOfElements", f"{nk+2*ng} {nj+2*ng} {ni+2*ng}")

        for coord, i in zip(["x", "y", "z"], [0, 1, 2]):
            X = blockElem.find("Geometry")[i]
            X.set("Dimensions", f"{nk+2*ng} {nj+2*ng} {ni+2*ng}")
            X.text = self.getGridFileH5Location(coord, nblki)

        self.gridElem.append(deepcopy(blockElem))

        return self.gridElem[-1]

    def getVarFileH5Location(self, varName, nrt, nblki):
        if self.lump:
            if self.animate:
                return f"q.{nrt:08d}.h5:/results_{nblki:06d}/{varName}"
            else:
                return f"q.h5:/results_{nblki:06d}/{varName}"
        else:
            if self.animate:
                return f"q.{nrt:08d}.{nblki:06d}.h5:/results_{nblki:06d}/{varName}"
            else:
                return f"q.{nblki:06d}.h5:/results_{nblki:06d}/{varName}"

    def getVarFileName(self, nrt, nblki):
        if self.lump:
            if self.animate:
                return f"q.{nrt:08d}.h5"
            else:
                return "q.h5"
        else:
            if self.animate:
                return f"q.{nrt:08d}.{nblki:06d}.h5"
            else:
                return f"q.{nblki:06d}.h5"

    def addScalarToBlockElem(self, blockElem, varName, nrt, nblki, ni, nj, nk, ng):
        attributeElem = deepcopy(self.scalarAttributeTemplate)
        attributeElem.set("Name", varName)

        dataItemElem = deepcopy(self.dataItemTemplate)
        dataItemElem.set("Dimensions", f"{nk+2*ng-1} {nj+2*ng-1} {ni+2*ng-1}")
        dataItemElem.text = self.getVarFileH5Location(varName, nrt, nblki)

        attributeElem.append(dataItemElem)
        blockElem.append(attributeElem)

    def addVectorToBlockElem(
        self, blockElem, vectorName, varNames, nrt, nblki, ni, nj, nk, ng
    ):
        attributeElem = deepcopy(self.vectorAttributeTemplate)
        attributeElem.set("Name", vectorName)
        functionElem = attributeElem.find("DataItem")
        functionElem.set("Dimensions", f"{nk+2*ng-1} {nj+2*ng-1} {ni+2*ng-1} 3")
        for varName in varNames:
            dataItemElem = deepcopy(self.dataItemTemplate)
            dataItemElem.set("Dimensions", f"{nk+2*ng-1} {nj+2*ng-1} {ni+2*ng-1}")
            dataItemElem.text = self.getVarFileH5Location(varName, nrt, nblki)

            functionElem.append(dataItemElem)

        blockElem.append(attributeElem)

    def getOutputName(self, nrt):
        if self.animate:
            outputName = f"q.{nrt:08d}.xmf"
        else:
            outputName = "q.xmf"
        return outputName


class arbitraryMetaData(restartMetaData):
    def __init__(self, arrayName, gridPath, precision, animate, lump, nrt=0, tme=0.0):
        super().__init__(gridPath, precision, animate, lump, nrt=0, tme=0.0)
        self.metaType = "arbitrary"
        self.arrayName = arrayName

    def getVarFileH5Location(self, varName, nrt, nblki):
        arrayName = self.arrayName
        if self.lump:
            if self.animate:
                return f"{arrayName}.{nrt:08d}.h5:/results_{nblki:06d}/{varName}"
            else:
                return f"{arrayName}.h5:/results_{nblki:06d}/{varName}"
        else:
            if self.animate:
                return f"{arrayName}.{nrt:08d}.{nblki:06d}.h5:/results_{nblki:06d}/{varName}"
            else:
                return f"{arrayName}.{nblki:06d}.h5:/results_{nblki:06d}/{varName}"

    def getVarFileName(self, nrt, nblki):
        arrayName = self.arrayName
        if self.lump:
            if self.animate:
                return f"{arrayName}.{nrt:08d}.h5"
            else:
                return f"{arrayName}.h5"
        else:
            if self.animate:
                return f"{arrayName}.{nrt:08d}.{nblki:06d}.h5"
            else:
                return f"{arrayName}.{nblki:06d}.h5"

    def getOutputName(self, nrt):
        if self.animate:
            outputName = f"{self.arrayName}.{nrt:08d}.xmf"
        else:
            outputName = f"{self.arrayName}.xmf"
        return outputName
